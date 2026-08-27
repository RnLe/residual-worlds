"""Shared value types crossing module boundaries.

Batched tensors use the last dimension for features throughout:
state ``[..., 4]`` as ``(q1, q2, qd1, qd2)``, action ``[..., 2]`` as
``(u1, u2)``, acceleration ``[..., 2]``. Public entry points validate
shape, dtype, and finiteness; inner rollout loops assume validated
tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

STATE_DIM = 4
ACTION_DIM = 2


@dataclass(frozen=True)
class ArmParameters:
    """Nominal rigid-body parameters of the planar two-link arm (SI units)."""

    link_lengths_m: tuple[float, float]
    com_lengths_m: tuple[float, float]
    masses_kg: tuple[float, float]
    inertias_kg_m2: tuple[float, float]
    viscous_nm_s_rad: tuple[float, float]
    gravity_m_s2: float
    torque_limit_nm: tuple[float, float]
    q_min_rad: tuple[float, float]
    q_max_rad: tuple[float, float]
    speed_limit_rad_s: tuple[float, float]


@dataclass(frozen=True)
class Scenario:
    """One frozen task instance shared across methods."""

    scenario_id: str
    bank: str
    index: int
    stratum_id: int
    target_order: tuple[int, int, int]
    obstacle_chord_index: int
    initial_state: tuple[float, float, float, float]
    targets_xy_m: tuple[tuple[float, float], ...]
    obstacle_xy_radius_m: tuple[float, float, float]
    timeout_steps: int


@dataclass(frozen=True)
class TaskState:
    """Deterministic task automaton state carried alongside the physical state."""

    target_index: int
    dwell_count: int
    previous_action: tuple[float, float]


@dataclass(frozen=True)
class PlanResult:
    """Outcome of one MPC planning call."""

    actions: torch.Tensor  # [execute_actions, 2] physical torques to issue
    latent_mean: torch.Tensor  # [knots, 2] final latent mean (warm-start carrier)
    predicted_states: torch.Tensor  # [horizon + 1, 4] ensemble-mean trajectory of the plan
    predicted_cost: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RolloutResult:
    """One completed closed-loop episode in the true target world."""

    evaluation_job_id: str
    scenario_id: str
    method_id: str
    states: np.ndarray  # [T + 1, 4]
    actions: np.ndarray  # [T, 2]
    realized_stage_costs: np.ndarray  # [T]
    target_index: np.ndarray  # [T + 1]
    dwell_count: np.ndarray  # [T + 1]
    mpc_call_index: np.ndarray  # [T]
    minimum_clearance_m: np.ndarray  # [T]
    events: tuple[dict[str, Any], ...]
    termination_reason: str
    success: bool


class DynamicsModel(Protocol):
    """A planner-facing model: continuous acceleration plus discrete step.

    The target simulator is *not* a ``DynamicsModel``; the evaluation
    harness owns it and only ever hands the controller one of these.
    """

    model_id: str

    def acceleration(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...


# Failure taxonomy, in fixed precedence order (first match wins).
FAILURE_CODES = (
    "TRAINING_FAILED",
    "NONFINITE_OR_MODEL_ERROR",
    "HARD_LIMIT_OR_SPEED",
    "OBSTACLE_COLLISION",
    "TIMEOUT_ZERO_TARGETS",
    "TIMEOUT_PARTIAL_TARGETS",
)

SUCCESS_CODE = "SUCCESS"

METHOD_IDS = ("nominal", "fitted_physics", "blackbox", "residual", "oracle")

# Public display name for the oracle condition; the internal ID stays short.
METHOD_DISPLAY_NAMES = {
    "nominal": "nominal physics",
    "fitted_physics": "fitted physics",
    "blackbox": "black box",
    "residual": "residual",
    "oracle": "exact-dynamics reference",
}
