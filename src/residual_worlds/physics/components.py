"""Shared mechanical component terms (public mechanics, not target secrets).

The point-payload terms live here rather than in the target-world module
because two sides legitimately need them: the hidden target simulator
composes them with friction/actuator mismatch, and the *fitted-physics*
baseline hypothesizes a payload as one of its five bounded parameters.
The equations are textbook mechanics; what stays private to the target
module is which values the hidden world actually uses.

For a point mass m_p at the end effector:

    M_p(q) = m_p J_e(q)^T J_e(q)          (tested against this identity)
    c_p    from beta_p = m_p l1 l2, h_p = -beta_p sin q2
    g_p    = m_p g [l1 cos q1 + l2 cos(q1+q2), l2 cos(q1+q2)]
"""

from __future__ import annotations

import torch

from residual_worlds.types import ArmParameters


def payload_mass_matrix(
    q: torch.Tensor, payload_kg: torch.Tensor | float, arm: ArmParameters
) -> torch.Tensor:
    l1, l2 = arm.link_lengths_m
    cos_q2 = torch.cos(q[..., 1])
    m_p = torch.as_tensor(payload_kg, dtype=q.dtype, device=q.device)
    m11 = m_p * (l1**2 + l2**2 + 2.0 * l1 * l2 * cos_q2)
    m12 = m_p * (l2**2 + l1 * l2 * cos_q2)
    m22 = m_p * l2**2 * torch.ones_like(cos_q2)
    row1 = torch.stack((m11, m12), dim=-1)
    row2 = torch.stack((m12, m22), dim=-1)
    return torch.stack((row1, row2), dim=-2)


def payload_coriolis(
    q: torch.Tensor, qd: torch.Tensor, payload_kg: torch.Tensor | float, arm: ArmParameters
) -> torch.Tensor:
    l1, l2 = arm.link_lengths_m
    m_p = torch.as_tensor(payload_kg, dtype=q.dtype, device=q.device)
    h = -m_p * l1 * l2 * torch.sin(q[..., 1])
    qd1, qd2 = qd[..., 0], qd[..., 1]
    return torch.stack((h * (2.0 * qd1 * qd2 + qd2**2), -h * qd1**2), dim=-1)


def payload_gravity(
    q: torch.Tensor, payload_kg: torch.Tensor | float, arm: ArmParameters
) -> torch.Tensor:
    l1, l2 = arm.link_lengths_m
    g = arm.gravity_m_s2
    m_p = torch.as_tensor(payload_kg, dtype=q.dtype, device=q.device)
    q1 = q[..., 0]
    q12 = q[..., 0] + q[..., 1]
    shared = m_p * g * l2 * torch.cos(q12)
    return torch.stack((m_p * g * l1 * torch.cos(q1) + shared, shared), dim=-1)
