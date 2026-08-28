"""Imagined-rollout engine: batched dynamics, task automaton, and cost.

For each candidate action sequence and each ensemble member, the
assigned model is rolled forward through the common integrator; every
transition's swept trace feeds the obstacle/joint barriers, the
per-candidate task automaton advances from that member's own predicted
kinematics, and costs accumulate under two masking rules:

* a candidate that safely completes the final dwell becomes absorbing:
  the completing transition is charged normally, then all later stage,
  safety, and terminal costs are zero and its state is held;
* the first invalid transition (swept collision, hard-limit crossing,
  non-finite state) is charged the finite invalid penalty exactly once,
  after which the candidate is terminated and masked -- safety failure
  takes precedence over completion on the same transition.

The same stage-cost function recomputes *realized* cost on true
executed trajectories, so predicted and realized costs are always
commensurable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from residual_worlds.config import CostWeights
from residual_worlds.models.base import Stepper
from residual_worlds.physics.integrators import AccelerationFn
from residual_worlds.physics.kinematics import end_effector_velocity
from residual_worlds.task.geometry import swept_transition_check
from residual_worlds.task.reaching import TaskRules, active_target_distance, advance_task
from residual_worlds.types import ArmParameters, Scenario


@dataclass(frozen=True)
class ScenarioTensors:
    """Scenario geometry as tensors on the planning device/dtype."""

    targets: torch.Tensor  # [3, 2]
    obstacle_center: torch.Tensor  # [2]
    obstacle_radius: float

    @staticmethod
    def from_scenario(
        scenario: Scenario, dtype: torch.dtype, device: torch.device | str = "cpu"
    ) -> ScenarioTensors:
        ox, oy, radius = scenario.obstacle_xy_radius_m
        return ScenarioTensors(
            targets=torch.tensor(scenario.targets_xy_m, dtype=dtype, device=device),
            obstacle_center=torch.tensor([ox, oy], dtype=dtype, device=device),
            obstacle_radius=radius,
        )


@dataclass(frozen=True)
class RolloutOutcome:
    """Batched result of imagining candidate futures (leading dims ``[...]``)."""

    total_cost: torch.Tensor  # [...]
    invalid: torch.Tensor  # [...] bool
    completed: torch.Tensor  # [...] bool
    states: torch.Tensor  # [..., H + 1, 4] predicted control-step states
    min_clearance: torch.Tensor  # [...] minimum swept clearance over active steps


def transition_stage_cost(
    endpoint_state: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    target_index: torch.Tensor,
    min_swept_clearance: torch.Tensor,
    min_joint_margin: torch.Tensor,
    targets: torch.Tensor,
    weights: CostWeights,
    rules: TaskRules,
    arm: ArmParameters,
) -> torch.Tensor:
    """Stage cost of one transition, batched over leading dimensions.

    ``endpoint_state`` is the transition's resulting state; barrier
    inputs summarize the full swept trace of the transition.
    """
    q, qd = endpoint_state[..., :2], endpoint_state[..., 2:]
    distance = active_target_distance(q, target_index, targets, arm)
    position_cost = weights.position * distance**2

    speed = torch.linalg.vector_norm(end_effector_velocity(q, qd, arm), dim=-1)
    near = (distance <= rules.near_target_velocity_radius_m) & (
        target_index < rules.target_count
    )
    velocity_cost = weights.near_target_velocity * speed**2 * near.to(speed.dtype)

    effort = weights.torque * torch.sum(action**2, dim=-1)
    smoothness = weights.torque_change * torch.sum((action - previous_action) ** 2, dim=-1)

    obstacle_hinge = torch.clamp(
        1.0 - min_swept_clearance / weights.obstacle_soft_margin_m, min=0.0
    )
    obstacle_cost = weights.obstacle_barrier * obstacle_hinge**2
    joint_hinge = torch.clamp(
        1.0 - min_joint_margin / weights.joint_soft_margin_rad, min=0.0
    )
    joint_cost = weights.joint_barrier * torch.sum(joint_hinge**2, dim=-1)

    total: torch.Tensor = (
        position_cost + velocity_cost + effort + smoothness + obstacle_cost + joint_cost
    )
    return total


def terminal_cost(
    state: torch.Tensor,
    target_index: torch.Tensor,
    targets: torch.Tensor,
    weights: CostWeights,
    rules: TaskRules,
    arm: ArmParameters,
) -> torch.Tensor:
    distance = active_target_distance(state[..., :2], target_index, targets, arm)
    remaining = (rules.target_count - target_index).clamp(min=0).to(state.dtype)
    return weights.terminal_position * distance**2 + weights.remaining_target * remaining


def imagine_rollout(
    acceleration: AccelerationFn,
    initial_state: torch.Tensor,
    initial_target_index: torch.Tensor,
    initial_dwell: torch.Tensor,
    previous_action: torch.Tensor,
    actions: torch.Tensor,
    scenario: ScenarioTensors,
    weights: CostWeights,
    rules: TaskRules,
    arm: ArmParameters,
    stepper: Stepper,
) -> RolloutOutcome:
    """Roll one model through candidate action sequences.

    Shapes: ``initial_state`` ``[..., 4]`` (broadcast to the candidate
    batch), ``actions`` ``[..., H, 2]``, automaton tensors ``[...]``.
    """
    horizon = actions.shape[-2]
    state = initial_state.expand(actions.shape[:-2] + (4,)).clone()
    target_index = initial_target_index.expand(actions.shape[:-2]).clone()
    dwell = initial_dwell.expand(actions.shape[:-2]).clone()
    previous = previous_action.expand(actions.shape[:-2] + (2,)).clone()

    total = torch.zeros(actions.shape[:-2], dtype=state.dtype, device=state.device)
    invalid = torch.zeros(actions.shape[:-2], dtype=torch.bool, device=state.device)
    completed = target_index >= rules.target_count
    min_clearance_overall = torch.full_like(total, float("inf"))
    trajectory = [state]

    substep_h = stepper.control_dt_s / stepper.substeps
    for h in range(horizon):
        action = actions[..., h, :]
        active = ~(invalid | completed)

        endpoints = stepper.substep_endpoints(acceleration, state, action)
        next_state = endpoints[..., -1, :]
        finite = torch.isfinite(endpoints).all(dim=-1).all(dim=-1)
        # Guard the geometry checks against non-finite values.
        safe_endpoints = torch.where(
            finite.unsqueeze(-1).unsqueeze(-1), endpoints, torch.zeros_like(endpoints)
        )
        check = swept_transition_check(
            safe_endpoints[..., :-1, :2],
            safe_endpoints[..., :-1, 2:],
            safe_endpoints[..., 1:, :2],
            safe_endpoints[..., 1:, 2:],
            substep_h,
            scenario.obstacle_center,
            scenario.obstacle_radius,
            rules.arm_safety_radius_m,
            rules.swept_inflation_m,
            arm,
        )
        min_clearance = check.min_clearance_m.min(dim=-1).values
        min_margin = check.min_joint_margin_rad.min(dim=-2).values
        newly_invalid = (
            ~finite | check.collision.any(dim=-1) | check.limit_violation.any(dim=-1)
        ) & active

        stage = transition_stage_cost(
            next_state,
            action,
            previous,
            target_index,
            min_clearance,
            min_margin,
            scenario.targets,
            weights,
            rules,
            arm,
        )
        # Invalidity replaces the stage cost with the one-time penalty;
        # completed/invalid candidates accrue nothing further.
        stage = torch.where(
            newly_invalid, torch.full_like(stage, weights.invalid_candidate), stage
        )
        total = total + stage * active.to(total.dtype)
        min_clearance_overall = torch.where(
            active & finite, torch.minimum(min_clearance_overall, min_clearance),
            min_clearance_overall,
        )

        # Advance the automaton only for candidates that stayed valid.
        advanced_index, advanced_dwell = advance_task(
            next_state[..., :2], next_state[..., 2:], target_index, dwell,
            scenario.targets, rules, arm,
        )
        progressing = active & ~newly_invalid
        target_index = torch.where(progressing, advanced_index, target_index)
        dwell = torch.where(progressing, advanced_dwell, dwell)
        newly_completed = progressing & (target_index >= rules.target_count)

        invalid = invalid | newly_invalid
        completed = completed | newly_completed
        # Progressing candidates take the transition (including the
        # completing one); newly invalid candidates hold their
        # pre-transition state so non-finite values never propagate.
        state = torch.where(progressing.unsqueeze(-1), next_state, state)
        previous = torch.where(progressing.unsqueeze(-1), action, previous)
        trajectory.append(state)

    still_active = ~(invalid | completed)
    total = total + terminal_cost(
        state, target_index, scenario.targets, weights, rules, arm
    ) * still_active.to(total.dtype)

    return RolloutOutcome(
        total_cost=total,
        invalid=invalid,
        completed=completed,
        states=torch.stack(trajectory, dim=-2),
        min_clearance=min_clearance_overall,
    )
