"""Collision geometry: distances, swept certification, tunneling."""

import numpy as np
import pytest
import torch

from residual_worlds.config import load_contract
from residual_worlds.paths import repository_root
from residual_worlds.task.geometry import (
    segment_circle_distance,
    swept_transition_check,
)

pytestmark = pytest.mark.scientific

CONTRACT = load_contract(repository_root() / "configs" / "experiment_contract.yaml")
ARM = CONTRACT.arm


def test_segment_circle_distance_known_cases() -> None:
    a = torch.tensor([0.0, 0.0], dtype=torch.float64)
    b = torch.tensor([1.0, 0.0], dtype=torch.float64)
    # Center above the middle of the segment.
    center = torch.tensor([0.5, 0.4], dtype=torch.float64)
    assert float(segment_circle_distance(a, b, center)) == pytest.approx(0.4)
    # Center beyond the end: distance to the endpoint.
    center = torch.tensor([1.5, 0.0], dtype=torch.float64)
    assert float(segment_circle_distance(a, b, center)) == pytest.approx(0.5)
    # Center exactly on the segment.
    center = torch.tensor([0.25, 0.0], dtype=torch.float64)
    assert float(segment_circle_distance(a, b, center)) == pytest.approx(0.0)
    # Degenerate segment (a == b): plain point distance.
    assert float(
        segment_circle_distance(a, a, torch.tensor([0.3, 0.4], dtype=torch.float64))
    ) == pytest.approx(0.5)


def _check(
    q_a: list[float],
    qd_a: list[float],
    q_b: list[float],
    qd_b: list[float],
    h: float,
    center: list[float],
    radius: float,
):
    return swept_transition_check(
        torch.tensor(q_a, dtype=torch.float64),
        torch.tensor(qd_a, dtype=torch.float64),
        torch.tensor(q_b, dtype=torch.float64),
        torch.tensor(qd_b, dtype=torch.float64),
        h,
        torch.tensor(center, dtype=torch.float64),
        radius,
        CONTRACT.task.arm_safety_radius_m,
        CONTRACT.task.swept_collision_inflation_m,
        ARM,
    )


def test_endpoint_only_check_would_miss_tunneling() -> None:
    # A fast sweep whose endpoints are clear but whose mid-path crosses
    # an obstacle placed at the arm tip's mid-trajectory. The shoulder
    # sweeps from 0.4 to 1.2 rad in one interval; the obstacle sits at
    # the tip position for q1 = 0.8.
    q_a, q_b = [0.4, 0.0], [1.2, 0.0]
    qd = [(q_b[0] - q_a[0]) / 0.05, 0.0]
    tip_mid = [
        float(np.cos(0.8)) * 1.0,
        float(np.sin(0.8)) * 1.0,
    ]
    result = _check(q_a, qd, q_b, qd, 0.05, tip_mid, 0.03)
    assert bool(result.collision)
    # Endpoints alone are clear of the same obstacle.
    endpoints_clear = _check(q_a, [0.0, 0.0], q_a, [0.0, 0.0], 0.05, tip_mid, 0.03)
    assert not bool(endpoints_clear.collision)


def test_stationary_clear_transition_is_certified() -> None:
    result = _check([0.9, -0.5], [0.0, 0.0], [0.9, -0.5], [0.0, 0.0], 0.05, [0.0, -0.8], 0.08)
    assert not bool(result.collision)
    assert not bool(result.limit_violation)
    assert float(result.min_clearance_m) > 0.3


def test_joint_margin_uses_exact_cubic_extrema() -> None:
    # Endpoints inside the limit but the interpolant overshoots past it:
    # equal positions with large opposite velocities bulge the cubic.
    q = [CONTRACT.arm.q_max_rad[0] - 0.01, 0.0]
    result = _check(q, [6.0, 0.0], q, [-6.0, 0.0], 0.05, [0.0, -0.9], 0.05)
    # The Hermite bulge rises ~ v*h/4 = 0.075 rad above the endpoints,
    # crossing the hard limit even though both endpoints respect it.
    assert bool(result.limit_violation)
    assert float(result.min_joint_margin_rad[0]) < 0.0


def test_velocity_limit_checked_along_trace() -> None:
    q_a, q_b = [0.5, 0.2], [0.55, 0.2]
    result = _check(q_a, [9.0, 0.0], q_b, [0.0, 0.0], 0.05, [0.0, -0.9], 0.05)
    assert bool(result.limit_violation)


def test_batched_shapes() -> None:
    batch = 7
    q_a = torch.rand(batch, 2, dtype=torch.float64) * 0.5 + 0.3
    qd = torch.zeros(batch, 2, dtype=torch.float64)
    result = swept_transition_check(
        q_a,
        qd,
        q_a + 0.01,
        qd,
        0.05,
        torch.tensor([0.0, -0.9], dtype=torch.float64),
        0.05,
        CONTRACT.task.arm_safety_radius_m,
        CONTRACT.task.swept_collision_inflation_m,
        ARM,
    )
    assert result.min_clearance_m.shape == (batch,)
    assert result.min_joint_margin_rad.shape == (batch, 2)
    assert result.collision.shape == (batch,)
