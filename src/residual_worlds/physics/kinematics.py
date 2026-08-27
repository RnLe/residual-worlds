"""Forward kinematics of the planar two-link arm.

Conventions (fixed throughout the project):

* the base joint sits at the origin of a vertical plane; gravity acts
  along -y;
* ``q1`` is the shoulder angle, measured counterclockwise from the
  positive x axis;
* ``q2`` is the elbow angle relative to link 1, so link 2 has absolute
  angle ``q1 + q2``.

Everything here is batched over arbitrary leading dimensions with the
last dimension holding coordinates, and differentiable, because the
same functions serve the simulator, the task cost, the renderer, and
the analysis. There is deliberately only one implementation of these
equations in the repository.
"""

from __future__ import annotations

import torch

from residual_worlds.types import ArmParameters


def elbow_position(q: torch.Tensor, arm: ArmParameters) -> torch.Tensor:
    """Position of the elbow joint, shape ``[..., 2]`` from ``q`` ``[..., 2]``."""
    l1 = arm.link_lengths_m[0]
    q1 = q[..., 0]
    return torch.stack((l1 * torch.cos(q1), l1 * torch.sin(q1)), dim=-1)


def end_effector_position(q: torch.Tensor, arm: ArmParameters) -> torch.Tensor:
    """End-effector position, shape ``[..., 2]``."""
    l1, l2 = arm.link_lengths_m
    q1 = q[..., 0]
    q12 = q[..., 0] + q[..., 1]
    x = l1 * torch.cos(q1) + l2 * torch.cos(q12)
    y = l1 * torch.sin(q1) + l2 * torch.sin(q12)
    return torch.stack((x, y), dim=-1)


def com_positions(q: torch.Tensor, arm: ArmParameters) -> tuple[torch.Tensor, torch.Tensor]:
    """Centers of mass of both links, each shape ``[..., 2]``."""
    l1 = arm.link_lengths_m[0]
    lc1, lc2 = arm.com_lengths_m
    q1 = q[..., 0]
    q12 = q[..., 0] + q[..., 1]
    p1 = torch.stack((lc1 * torch.cos(q1), lc1 * torch.sin(q1)), dim=-1)
    p2 = torch.stack(
        (l1 * torch.cos(q1) + lc2 * torch.cos(q12), l1 * torch.sin(q1) + lc2 * torch.sin(q12)),
        dim=-1,
    )
    return p1, p2


def end_effector_jacobian(q: torch.Tensor, arm: ArmParameters) -> torch.Tensor:
    """Analytic Jacobian d p_e / d q, shape ``[..., 2, 2]``."""
    l1, l2 = arm.link_lengths_m
    q1 = q[..., 0]
    q12 = q[..., 0] + q[..., 1]
    s1, c1 = torch.sin(q1), torch.cos(q1)
    s12, c12 = torch.sin(q12), torch.cos(q12)
    row_x = torch.stack((-l1 * s1 - l2 * s12, -l2 * s12), dim=-1)
    row_y = torch.stack((l1 * c1 + l2 * c12, l2 * c12), dim=-1)
    return torch.stack((row_x, row_y), dim=-2)


def end_effector_velocity(
    q: torch.Tensor, qd: torch.Tensor, arm: ArmParameters
) -> torch.Tensor:
    """Cartesian end-effector velocity ``J(q) qd``, shape ``[..., 2]``."""
    jacobian = end_effector_jacobian(q, arm)
    return torch.einsum("...ij,...j->...i", jacobian, qd)


def link_segments(
    q: torch.Tensor, arm: ArmParameters
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Segment endpoints (base, elbow, end effector) for collision geometry.

    Returns three ``[..., 2]`` tensors; link 1 spans base->elbow and
    link 2 spans elbow->end effector.
    """
    base = torch.zeros_like(q)
    return base, elbow_position(q, arm), end_effector_position(q, arm)
