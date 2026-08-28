"""Receding-horizon controller: warm-started CEM over the assigned model.

Every method uses this identical controller; only the acceleration
function(s) differ. Ensemble scoring follows the frozen convention:
each candidate is rolled through every member, the scalar ranking key
is (any member invalid, arithmetic mean of member costs, candidate
index), and the final latent mean is valid only if every member's
rollout is valid.

Primitive CEM noise is drawn per planning call from a fresh CPU
``PCG64DXSM`` generator seeded by a method-free namespace, as float64,
hashed in canonical little-endian bytes before casting to the planning
dtype -- so paired methods demonstrably receive identical randomness
while their updated distributions are free to diverge.

Warm starting lives entirely in latent space: expand the previous knot
mean to the horizon, shift left by the number of executed actions,
repeat the final latent action into the vacated tail, and compress back
to knots by averaging consecutive pairs. The latent standard deviation
resets to its initial value at every call; only within-call elite
updates shrink it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from residual_worlds.config import CostWeights, PlanningConfig
from residual_worlds.models.base import Stepper
from residual_worlds.physics.integrators import AccelerationFn
from residual_worlds.planning.cem import (
    CEMResult,
    CEMSettings,
    cem_optimize,
    expand_knots,
    latent_to_action,
)
from residual_worlds.planning.costs import RolloutOutcome, ScenarioTensors, imagine_rollout
from residual_worlds.seeds import numpy_generator
from residual_worlds.task.reaching import TaskRules
from residual_worlds.types import ArmParameters, PlanResult, TaskState

NoiseFn = Callable[[int], tuple[torch.Tensor, str]]


def cem_base_noise(
    root_seed: int, tokens: Sequence[str | int], shape: tuple[int, int, int, int]
) -> tuple[torch.Tensor, str]:
    """Method-free primitive noise for one planning call, with its hash."""
    rng = numpy_generator(root_seed, *tokens)
    values = rng.standard_normal(size=shape, dtype=np.float64)
    digest = hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()
    return torch.from_numpy(values), digest


def make_noise_fn(
    root_seed: int, namespace: Sequence[str | int], shape: tuple[int, int, int, int]
) -> NoiseFn:
    """Bind a call namespace; the MPC step index is appended per call."""

    def draw(mpc_step: int) -> tuple[torch.Tensor, str]:
        return cem_base_noise(root_seed, (*namespace, mpc_step), shape)

    return draw


@dataclass(frozen=True)
class ControllerSettings:
    horizon_steps: int
    action_knots: int
    execute_actions_per_plan: int
    replan_every_steps: int
    cem: CEMSettings

    @staticmethod
    def from_planning(planning: PlanningConfig) -> ControllerSettings:
        return ControllerSettings(
            horizon_steps=planning.horizon_steps,
            action_knots=planning.action_knots,
            execute_actions_per_plan=planning.execute_actions_per_plan,
            replan_every_steps=planning.replan_every_steps,
            cem=CEMSettings.from_planning(planning),
        )


class MPCController:
    def __init__(
        self,
        members: Sequence[AccelerationFn],
        scenario_tensors: ScenarioTensors,
        weights: CostWeights,
        rules: TaskRules,
        arm: ArmParameters,
        stepper: Stepper,
        settings: ControllerSettings,
        noise_fn: NoiseFn,
    ) -> None:
        if not members:
            raise ValueError("the controller needs at least one model member")
        self._members = tuple(members)
        self._scenario = scenario_tensors
        self._weights = weights
        self._rules = rules
        self._arm = arm
        self._stepper = stepper
        self._settings = settings
        self._noise_fn = noise_fn
        self._torque_limit = torch.tensor(arm.torque_limit_nm, dtype=torch.float32)
        self._latent_mean = torch.zeros(settings.action_knots, 2, dtype=torch.float32)
        self._last_outcomes: list[RolloutOutcome] | None = None

    def reset(self) -> None:
        """Zero the warm start (episode start, or after a model change)."""
        self._latent_mean = torch.zeros_like(self._latent_mean)

    def _warm_start_shift(self) -> torch.Tensor:
        settings = self._settings
        expanded = expand_knots(self._latent_mean, settings.horizon_steps)
        shift = settings.execute_actions_per_plan
        tail = expanded[-1:].expand(shift, 2)
        shifted = torch.cat((expanded[shift:], tail), dim=0)
        # Compress back to knots by averaging consecutive groups.
        repeat = settings.horizon_steps // settings.action_knots
        compressed = shifted.reshape(settings.action_knots, repeat, 2).mean(dim=1)
        return compressed

    def _score(
        self, latent_candidates: torch.Tensor, state: torch.Tensor, task: TaskState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean member cost and any-member-invalid for latent candidates [K, knots, 2]."""
        actions = latent_to_action(
            expand_knots(latent_candidates, self._settings.horizon_steps),
            self._torque_limit,
        )
        count = actions.shape[0]
        index = torch.full((count,), task.target_index, dtype=torch.long)
        dwell = torch.full((count,), task.dwell_count, dtype=torch.long)
        previous = torch.tensor(task.previous_action, dtype=torch.float32).expand(count, 2)
        outcomes = [
            imagine_rollout(
                member,
                state.to(torch.float32),
                index,
                dwell,
                previous,
                actions,
                self._scenario,
                self._weights,
                self._rules,
                self._arm,
                self._stepper,
            )
            for member in self._members
        ]
        self._last_outcomes = outcomes
        member_costs = torch.stack([outcome.total_cost for outcome in outcomes], dim=0)
        member_invalid = torch.stack([outcome.invalid for outcome in outcomes], dim=0)
        return member_costs.mean(dim=0), member_invalid.any(dim=0)

    @torch.no_grad()
    def plan(self, state: torch.Tensor, task: TaskState, mpc_step: int) -> PlanResult:
        settings = self._settings
        base_noise, noise_digest = self._noise_fn(mpc_step)
        initial_mean = self._warm_start_shift()

        result: CEMResult = cem_optimize(
            lambda latents: self._score(latents, state, task),
            initial_mean,
            base_noise,
            settings.cem,
        )
        self._latent_mean = result.final_mean.clone()

        final_actions = latent_to_action(
            expand_knots(result.final_mean, settings.horizon_steps), self._torque_limit
        )
        # The scorer's last invocation was the final mean (batch of one);
        # reuse its rollouts for the predicted trajectory and diagnostics.
        assert self._last_outcomes is not None
        predicted_states = torch.stack(
            [outcome.states[0] for outcome in self._last_outcomes], dim=0
        ).mean(dim=0)
        plan_nonfinite = not (
            bool(torch.isfinite(final_actions).all())
            and all(bool(torch.isfinite(o.total_cost[0])) for o in self._last_outcomes)
        )

        diagnostics = {
            "noise_sha256": noise_digest,
            "best_cost_by_iteration": result.best_cost_by_iteration,
            "invalid_fraction_by_iteration": result.invalid_fraction_by_iteration,
            "elite_indices": result.elite_indices,
            "final_mean_invalid": result.final_mean_invalid,
            "plan_nonfinite": plan_nonfinite,
            "predicted_min_clearance": float(
                min(float(o.min_clearance[0]) for o in self._last_outcomes)
            ),
        }
        return PlanResult(
            actions=final_actions[: settings.execute_actions_per_plan],
            latent_mean=result.final_mean,
            predicted_states=predicted_states,
            predicted_cost=result.final_mean_cost,
            diagnostics=diagnostics,
        )
