"""Collision geometry: segment clearance and swept-transition checks.

The obstacle is a circle; each arm link is a segment with safety radius
``r_a``. Signed clearance of one configuration is

    d(q) = min_link dist(segment, center) - (r_o + r_a),

and a collision is ``d <= 0``.

Checking only control-step endpoints would let a fast link tunnel
through the obstacle, and the intermediate RK4 stage states are
derivative probes, not an ordered path, so they must never be chained.
Instead, between each pair of *accepted substep endpoints* the joint
path is approximated by the cubic Hermite interpolant defined by the
endpoint positions, velocities, and substep duration. For that cubic:

* each joint's exact excursion (max minus min over the interval) comes
  from the endpoints plus the real roots of the quadratic derivative;
* a conservative bound for the displacement of any point on link 1 is
  ``l1 * dq1`` and on link 2 ``(l1 + l2) * dq1 + l2 * dq2``;
* the interpolant is sampled densely enough that consecutive samples
  move by at most the frozen resolution (2 mm), and every sample is
  checked against the obstacle inflated by that same resolution -- a
  sample set that clears the inflated obstacle certifies the whole
  interval under this approximation.

Joint-limit margins use the exact cubic extrema (no sampling); velocity
extrema of the Hermite are checked at the endpoints and the interior
vertex of its quadratic derivative.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from residual_worlds.types import ArmParameters


@dataclass(frozen=True)
class SweptCheckResult:
    """Batched result of a swept-transition safety check (shapes ``[...]``)."""

    min_clearance_m: torch.Tensor  # minimum signed obstacle clearance along the trace
    min_joint_margin_rad: torch.Tensor  # [..., 2] minimum signed hard-limit margin
    max_abs_velocity_rad_s: torch.Tensor  # [..., 2] maximum |qd| along the trace
    collision: torch.Tensor  # bool: trace cannot be certified collision-free
    limit_violation: torch.Tensor  # bool: joint angle or speed limit crossed


def segment_circle_distance(
    a: torch.Tensor, b: torch.Tensor, center: torch.Tensor
) -> torch.Tensor:
    """Distance from ``center`` to segment ``a``-``b``; all ``[..., 2]``."""
    ab = b - a
    ac = center - a
    denominator = torch.sum(ab * ab, dim=-1).clamp_min(1e-30)
    t = (torch.sum(ac * ab, dim=-1) / denominator).clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    distance: torch.Tensor = torch.linalg.vector_norm(center - closest, dim=-1)
    return distance


def arm_clearance_with_radius(
    q: torch.Tensor,
    obstacle_center: torch.Tensor,
    obstacle_radius: torch.Tensor | float,
    arm_safety_radius_m: float,
    arm: ArmParameters,
) -> torch.Tensor:
    """Signed clearance of both links at configurations ``q`` ``[..., 2]``."""
    from residual_worlds.physics.kinematics import link_segments

    base, elbow, tip = link_segments(q, arm)
    center = torch.as_tensor(obstacle_center, dtype=q.dtype, device=q.device)
    center = center.expand(q.shape[:-1] + (2,))
    d1 = segment_circle_distance(base, elbow, center)
    d2 = segment_circle_distance(elbow, tip, center)
    radius = torch.as_tensor(obstacle_radius, dtype=q.dtype, device=q.device)
    return torch.minimum(d1, d2) - (radius + arm_safety_radius_m)


def _hermite_coefficients(
    q_a: torch.Tensor, qd_a: torch.Tensor, q_b: torch.Tensor, qd_b: torch.Tensor, h: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cubic Hermite q(s) = c0 + c1 s + c2 s^2 + c3 s^3 on s in [0, 1]."""
    v_a = qd_a * h
    v_b = qd_b * h
    c0 = q_a
    c1 = v_a
    c2 = 3.0 * (q_b - q_a) - 2.0 * v_a - v_b
    c3 = 2.0 * (q_a - q_b) + v_a + v_b
    return c0, c1, c2, c3


def _cubic_extrema(
    c0: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact min and max of the cubic on [0, 1], per batch element and joint."""

    def evaluate(s: torch.Tensor) -> torch.Tensor:
        return c0 + s * (c1 + s * (c2 + s * c3))

    ones = torch.ones_like(c0)
    candidates = [evaluate(torch.zeros_like(c0)), evaluate(ones)]
    # Roots of q'(s) = c1 + 2 c2 s + 3 c3 s^2.
    a = 3.0 * c3
    b = 2.0 * c2
    c = c1
    # Quadratic case (|a| tiny handled by the linear fallback).
    discriminant = b * b - 4.0 * a * c
    sqrt_disc = torch.sqrt(discriminant.clamp_min(0.0))
    safe_a = torch.where(torch.abs(a) > 1e-30, a, torch.ones_like(a))
    for sign in (1.0, -1.0):
        root = (-b + sign * sqrt_disc) / (2.0 * safe_a)
        valid = (torch.abs(a) > 1e-30) & (discriminant >= 0.0) & (root > 0.0) & (root < 1.0)
        clamped = root.clamp(0.0, 1.0)
        value = evaluate(clamped)
        candidates.append(torch.where(valid, value, candidates[0]))
    # Linear-derivative fallback: root of b s + c = 0.
    safe_b = torch.where(torch.abs(b) > 1e-30, b, torch.ones_like(b))
    linear_root = -c / safe_b
    linear_valid = (torch.abs(a) <= 1e-30) & (linear_root > 0.0) & (linear_root < 1.0)
    candidates.append(
        torch.where(linear_valid, evaluate(linear_root.clamp(0.0, 1.0)), candidates[0])
    )
    stacked = torch.stack(candidates, dim=0)
    return stacked.min(dim=0).values, stacked.max(dim=0).values


def swept_transition_check(
    q_a: torch.Tensor,
    qd_a: torch.Tensor,
    q_b: torch.Tensor,
    qd_b: torch.Tensor,
    h: float,
    obstacle_center: torch.Tensor,
    obstacle_radius: torch.Tensor | float,
    arm_safety_radius_m: float,
    inflation_m: float,
    arm: ArmParameters,
    max_samples: int = 256,
) -> SweptCheckResult:
    """Certify one substep interval; all inputs batched ``[..., 2]``."""
    c0, c1, c2, c3 = _hermite_coefficients(q_a, qd_a, q_b, qd_b, h)
    q_min_traj, q_max_traj = _cubic_extrema(c0, c1, c2, c3)
    excursion = q_max_traj - q_min_traj  # [..., 2]

    l1, l2 = arm.link_lengths_m
    # Conservative whole-arm displacement bound over the interval.
    displacement_bound = (l1 + l2) * excursion[..., 0] + l2 * excursion[..., 1]
    needed = torch.ceil(displacement_bound / inflation_m).clamp(min=1.0)
    samples = int(min(max_samples, float(needed.max())) if needed.numel() else 1)
    samples = max(samples, 1)

    # Sample the Hermite path uniformly; consecutive samples then move by
    # at most ~displacement_bound / samples <= inflation for the worst
    # batch element (clamped by max_samples as a hard cap).
    s = torch.linspace(0.0, 1.0, samples + 1, dtype=q_a.dtype, device=q_a.device)
    shape = q_a.shape[:-1]
    s = s.view((samples + 1,) + (1,) * (len(shape) + 1))
    q_path = (
        c0.unsqueeze(0)
        + s * (c1.unsqueeze(0) + s * (c2.unsqueeze(0) + s * c3.unsqueeze(0)))
    )  # [samples + 1, ..., 2]
    clearance = arm_clearance_with_radius(
        q_path, obstacle_center, obstacle_radius, arm_safety_radius_m, arm
    )
    min_clearance = clearance.min(dim=0).values

    # A trace is certified collision-free only if every sample clears the
    # obstacle inflated by the sampling resolution.
    collision = min_clearance <= inflation_m

    q_lower = torch.as_tensor(arm.q_min_rad, dtype=q_a.dtype, device=q_a.device)
    q_upper = torch.as_tensor(arm.q_max_rad, dtype=q_a.dtype, device=q_a.device)
    margin = torch.minimum(q_min_traj - q_lower, q_upper - q_max_traj)

    # Velocity extrema of the Hermite: qd(s) = (c1 + 2 c2 s + 3 c3 s^2)/h;
    # check endpoints and the interior vertex of the quadratic.
    def velocity(sv: torch.Tensor) -> torch.Tensor:
        return (c1 + sv * (2.0 * c2 + sv * 3.0 * c3)) / max(h, 1e-30)

    vertex = torch.where(
        torch.abs(c3) > 1e-30, -c2 / (3.0 * c3), torch.zeros_like(c2)
    ).clamp(0.0, 1.0)
    speeds = torch.stack(
        [
            torch.abs(velocity(torch.zeros_like(c1))),
            torch.abs(velocity(torch.ones_like(c1))),
            torch.abs(velocity(vertex)),
        ],
        dim=0,
    )
    max_abs_velocity = speeds.max(dim=0).values
    speed_limit = torch.as_tensor(arm.speed_limit_rad_s, dtype=q_a.dtype, device=q_a.device)
    limit_violation = ((margin <= 0.0) | (max_abs_velocity > speed_limit)).any(dim=-1)

    return SweptCheckResult(
        min_clearance_m=min_clearance,
        min_joint_margin_rad=margin,
        max_abs_velocity_rad_s=max_abs_velocity,
        collision=collision,
        limit_violation=limit_violation,
    )
