"""Ordered three-target reaching task: automaton, rules, and truth environment.

Task rules (identical for the true episode and for imagined planner
rollouts): a target is completed after ``dwell_steps`` consecutive
control-step endpoints within ``target_radius`` of the active target
while the end-effector speed stays below ``speed_threshold``. Completion
activates the next target; the state after the final target is absorbing
(``target_index == target_count``).

Event precedence for one executed transition is fixed:

1. numerical or safety failure (non-finite state, hard joint/speed
   limit crossing, swept-trace collision) terminates the episode;
2. otherwise the dwell/target automaton advances, and completing the
   final dwell terminates the episode successfully -- even on the
   transition into the very last allowed step;
3. otherwise, after the ``timeout_steps``-th executed transition, the
   episode is truncated (a time limit, not a physical event).

The environment simulates truth in float64 with the frozen RK4 substep
count and receives its acceleration function from the evaluation
harness -- this module never constructs target worlds itself, so hidden
world parameters cannot leak through the task layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
import torch

from residual_worlds.config import TaskConfig
from residual_worlds.physics.integrators import AccelerationFn, rk4_substep_endpoints
from residual_worlds.physics.kinematics import end_effector_position, end_effector_velocity
from residual_worlds.task.geometry import swept_transition_check
from residual_worlds.types import ArmParameters, Scenario


@dataclass(frozen=True)
class TaskRules:
    """The subset of task configuration the automaton needs (all frozen)."""

    target_count: int
    target_radius_m: float
    target_speed_threshold_m_s: float
    near_target_velocity_radius_m: float
    target_dwell_steps: int
    arm_safety_radius_m: float
    swept_inflation_m: float

    @staticmethod
    def from_config(task: TaskConfig) -> TaskRules:
        return TaskRules(
            target_count=task.target_count,
            target_radius_m=task.target_radius_m,
            target_speed_threshold_m_s=task.target_speed_threshold_m_s,
            near_target_velocity_radius_m=task.near_target_velocity_radius_m,
            target_dwell_steps=task.target_dwell_steps,
            arm_safety_radius_m=task.arm_safety_radius_m,
            swept_inflation_m=task.swept_collision_inflation_m,
        )


def active_target_distance(
    q: torch.Tensor,
    target_index: torch.Tensor,
    targets: torch.Tensor,
    arm: ArmParameters,
) -> torch.Tensor:
    """Distance from the end effector to the active target; 0 when complete.

    ``q`` is ``[..., 2]``, ``target_index`` ``[...]`` (long), ``targets``
    ``[..., T, 2]`` broadcastable against the batch.
    """
    position = end_effector_position(q, arm)
    count = targets.shape[-2]
    broadcast = torch.broadcast_to(targets, target_index.shape + (count, 2))
    clamped = target_index.clamp(max=count - 1)
    gathered = torch.take_along_dim(
        broadcast, clamped.unsqueeze(-1).unsqueeze(-1).expand(*clamped.shape, 1, 2), dim=-2
    ).squeeze(-2)
    distance = torch.linalg.vector_norm(position - gathered, dim=-1)
    return torch.where(target_index >= count, torch.zeros_like(distance), distance)


def advance_task(
    q: torch.Tensor,
    qd: torch.Tensor,
    target_index: torch.Tensor,
    dwell_count: torch.Tensor,
    targets: torch.Tensor,
    rules: TaskRules,
    arm: ArmParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the dwell/target automaton at one accepted control endpoint.

    Batched over arbitrary leading dimensions; the completed state
    (``target_index == target_count``) is absorbing.
    """
    active = target_index < rules.target_count
    distance = active_target_distance(q, target_index, targets, arm)
    speed = torch.linalg.vector_norm(end_effector_velocity(q, qd, arm), dim=-1)
    inside = (distance <= rules.target_radius_m) & (speed <= rules.target_speed_threshold_m_s)
    new_dwell = torch.where(inside & active, dwell_count + 1, torch.zeros_like(dwell_count))
    completed = new_dwell >= rules.target_dwell_steps
    new_index = torch.where(completed & active, target_index + 1, target_index)
    new_dwell = torch.where(completed | ~active, torch.zeros_like(new_dwell), new_dwell)
    return new_index, new_dwell


class TrueArmEnv(gymnasium.Env[np.ndarray, np.ndarray]):
    """The true target-world episode. Observation is the exact state.

    The acceleration function is supplied by the evaluation harness;
    this class never sees world parameters, only the transition law.
    Rewards are always zero -- performance is judged by the recorded
    outcome, not by a shaped return.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        acceleration: AccelerationFn,
        scenario: Scenario,
        rules: TaskRules,
        arm: ArmParameters,
        control_dt_s: float,
        substeps: int,
    ) -> None:
        self._acceleration = acceleration
        self._scenario = scenario
        self._rules = rules
        self._arm = arm
        self._dt = control_dt_s
        self._substeps = substeps
        limit = np.array(arm.torque_limit_nm, dtype=np.float64)
        self.action_space = gymnasium.spaces.Box(-limit, limit, dtype=np.float64)
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, shape=(4,), dtype=np.float64
        )
        self._state = torch.zeros(4, dtype=torch.float64)
        self._target_index = 0
        self._dwell = 0
        self._executed_steps = 0
        self._targets = torch.tensor(scenario.targets_xy_m, dtype=torch.float64)
        ox, oy, orad = scenario.obstacle_xy_radius_m
        self._obstacle_center = torch.tensor([ox, oy], dtype=torch.float64)
        self._obstacle_radius = orad

    @property
    def state(self) -> np.ndarray:
        return self._state.numpy().copy()

    @property
    def target_index(self) -> int:
        return self._target_index

    @property
    def dwell_count(self) -> int:
        return self._dwell

    @property
    def executed_steps(self) -> int:
        return self._executed_steps

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = torch.tensor(self._scenario.initial_state, dtype=torch.float64)
        self._target_index = 0
        self._dwell = 0
        self._executed_steps = 0
        return self.state, {"scenario_id": self._scenario.scenario_id}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        rules = self._rules
        command = torch.as_tensor(action, dtype=torch.float64).clamp(
            min=torch.tensor([-v for v in self._arm.torque_limit_nm], dtype=torch.float64),
            max=torch.tensor(self._arm.torque_limit_nm, dtype=torch.float64),
        )
        endpoints = rk4_substep_endpoints(
            self._acceleration, self._state, command, self._dt, self._substeps
        )  # [substeps + 1, 4]
        info: dict[str, Any] = {"events": []}

        # 1. Numerical failure has the highest precedence.
        if not bool(torch.isfinite(endpoints).all()):
            self._executed_steps += 1
            info["reason"] = "NONFINITE_OR_MODEL_ERROR"
            info["min_clearance_m"] = float("nan")
            return self.state, 0.0, True, False, info

        h = self._dt / self._substeps
        check = swept_transition_check(
            endpoints[:-1, :2],
            endpoints[:-1, 2:],
            endpoints[1:, :2],
            endpoints[1:, 2:],
            h,
            self._obstacle_center,
            self._obstacle_radius,
            rules.arm_safety_radius_m,
            rules.swept_inflation_m,
            self._arm,
        )
        min_clearance = float(check.min_clearance_m.min())
        info["min_clearance_m"] = min_clearance
        margins = check.min_joint_margin_rad.min(dim=0).values
        info["min_joint_margin_rad"] = (float(margins[0]), float(margins[1]))
        self._state = endpoints[-1]
        self._executed_steps += 1

        if bool(check.limit_violation.any()):
            info["reason"] = "HARD_LIMIT_OR_SPEED"
            return self.state, 0.0, True, False, info
        if bool(check.collision.any()):
            info["reason"] = "OBSTACLE_COLLISION"
            return self.state, 0.0, True, False, info

        # 2. Task automaton and success.
        index = torch.tensor(self._target_index)
        dwell = torch.tensor(self._dwell)
        new_index, new_dwell = advance_task(
            self._state[:2], self._state[2:], index, dwell, self._targets, rules, self._arm
        )
        if int(new_index) > self._target_index:
            info["events"].append(
                {"type": "target_completed", "target": self._target_index,
                 "step": self._executed_steps}
            )
        self._target_index = int(new_index)
        self._dwell = int(new_dwell)
        if self._target_index >= rules.target_count:
            info["reason"] = "SUCCESS"
            return self.state, 0.0, True, False, info

        # 3. Timeout is truncation, never a physical event.
        if self._executed_steps >= self._scenario.timeout_steps:
            info["reason"] = (
                "TIMEOUT_PARTIAL_TARGETS" if self._target_index > 0 else "TIMEOUT_ZERO_TARGETS"
            )
            return self.state, 0.0, False, True, info

        return self.state, 0.0, False, False, info
