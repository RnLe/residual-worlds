"""Nominal rigid-body dynamics of the two-link arm.

The manipulator equation is

    M0(q) qdd + c0(q, qd) + g0(q) + B0 qd = u,

with configuration-dependent inertia M0, Coriolis/centrifugal vector
c0, gravity torque g0 (angles measured from the positive horizontal
axis, so gravity terms carry cosines), and diagonal viscous damping B0.

Derivation notes: with beta = m2 l1 lc2 and h(q2) = -beta sin(q2),

    M11 = I1 + I2 + m1 lc1^2 + m2 (l1^2 + lc2^2 + 2 l1 lc2 cos q2)
    M12 = M21 = I2 + m2 (lc2^2 + l1 lc2 cos q2)
    M22 = I2 + m2 lc2^2
    c   = [ h (2 qd1 qd2 + qd2^2), -h qd1^2 ]
    V   = g [ (m1 lc1 + m2 l1) sin q1 + m2 lc2 sin(q1 + q2) ]
    g0  = grad_q V.

The acceleration is obtained by solving the 2x2 linear system, never by
forming an explicit inverse. A scalar float64 NumPy reference of the
same equations lives at the bottom of this module purely as an
independent cross-check for the batched implementation.
"""

from __future__ import annotations

import numpy as np
import torch

from residual_worlds.types import ArmParameters


def mass_matrix(q: torch.Tensor, arm: ArmParameters) -> torch.Tensor:
    """Nominal inertia matrix M0(q), shape ``[..., 2, 2]``."""
    l1, _ = arm.link_lengths_m
    lc1, lc2 = arm.com_lengths_m
    m1, m2 = arm.masses_kg
    i1, i2 = arm.inertias_kg_m2
    cos_q2 = torch.cos(q[..., 1])
    m11 = i1 + i2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * cos_q2)
    m12 = i2 + m2 * (lc2**2 + l1 * lc2 * cos_q2)
    m22 = torch.full_like(cos_q2, i2 + m2 * lc2**2)
    row1 = torch.stack((m11, m12), dim=-1)
    row2 = torch.stack((m12, m22), dim=-1)
    return torch.stack((row1, row2), dim=-2)


def coriolis_vector(q: torch.Tensor, qd: torch.Tensor, arm: ArmParameters) -> torch.Tensor:
    """Coriolis/centrifugal torque vector c0(q, qd), shape ``[..., 2]``."""
    l1, _ = arm.link_lengths_m
    _, lc2 = arm.com_lengths_m
    _, m2 = arm.masses_kg
    beta = m2 * l1 * lc2
    h = -beta * torch.sin(q[..., 1])
    qd1, qd2 = qd[..., 0], qd[..., 1]
    c1 = h * (2.0 * qd1 * qd2 + qd2**2)
    c2 = -h * qd1**2
    return torch.stack((c1, c2), dim=-1)


def gravity_vector(q: torch.Tensor, arm: ArmParameters) -> torch.Tensor:
    """Generalized gravity torque g0(q) = grad V, shape ``[..., 2]``."""
    l1, _ = arm.link_lengths_m
    lc1, lc2 = arm.com_lengths_m
    m1, m2 = arm.masses_kg
    g = arm.gravity_m_s2
    q1 = q[..., 0]
    q12 = q[..., 0] + q[..., 1]
    shared = m2 * lc2 * g * torch.cos(q12)
    g1 = (m1 * lc1 + m2 * l1) * g * torch.cos(q1) + shared
    return torch.stack((g1, shared), dim=-1)


def potential_energy(q: torch.Tensor, arm: ArmParameters) -> torch.Tensor:
    """Gravitational potential V(q), shape ``[...]`` (zero at horizontal COMs)."""
    l1, _ = arm.link_lengths_m
    lc1, lc2 = arm.com_lengths_m
    m1, m2 = arm.masses_kg
    g = arm.gravity_m_s2
    return g * (
        (m1 * lc1 + m2 * l1) * torch.sin(q[..., 0])
        + m2 * lc2 * torch.sin(q[..., 0] + q[..., 1])
    )


def kinetic_energy(q: torch.Tensor, qd: torch.Tensor, arm: ArmParameters) -> torch.Tensor:
    """Kinetic energy 0.5 qd^T M(q) qd, shape ``[...]``."""
    m = mass_matrix(q, arm)
    return 0.5 * torch.einsum("...i,...ij,...j->...", qd, m, qd)


def damping_torque(qd: torch.Tensor, arm: ArmParameters) -> torch.Tensor:
    """Nominal viscous damping B0 qd, shape ``[..., 2]``."""
    b = torch.as_tensor(arm.viscous_nm_s_rad, dtype=qd.dtype, device=qd.device)
    return b * qd


def acceleration(
    q: torch.Tensor, qd: torch.Tensor, u: torch.Tensor, arm: ArmParameters
) -> torch.Tensor:
    """Nominal joint acceleration f0(q, qd, u), shape ``[..., 2]``."""
    rhs = u - coriolis_vector(q, qd, arm) - gravity_vector(q, arm) - damping_torque(qd, arm)
    m = mass_matrix(q, arm)
    solution: torch.Tensor = torch.linalg.solve(m, rhs.unsqueeze(-1))
    return solution.squeeze(-1)


def state_acceleration(
    state: torch.Tensor, action: torch.Tensor, arm: ArmParameters
) -> torch.Tensor:
    """Acceleration from a packed state ``[..., 4]`` and action ``[..., 2]``."""
    return acceleration(state[..., :2], state[..., 2:], action, arm)


# ---------------------------------------------------------------------------
# Independent scalar reference (float64 NumPy). Used only by tests and the
# verification suite to cross-check the batched implementation above.


def acceleration_reference_numpy(
    q: np.ndarray, qd: np.ndarray, u: np.ndarray, arm: ArmParameters
) -> np.ndarray:
    l1, _ = arm.link_lengths_m
    lc1, lc2 = arm.com_lengths_m
    m1, m2 = arm.masses_kg
    i1, i2 = arm.inertias_kg_m2
    b1, b2 = arm.viscous_nm_s_rad
    g = arm.gravity_m_s2

    cos_q2 = float(np.cos(q[1]))
    m11 = i1 + i2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * cos_q2)
    m12 = i2 + m2 * (lc2**2 + l1 * lc2 * cos_q2)
    m22 = i2 + m2 * lc2**2
    mass = np.array([[m11, m12], [m12, m22]], dtype=np.float64)

    h = -m2 * l1 * lc2 * float(np.sin(q[1]))
    coriolis = np.array(
        [h * (2.0 * qd[0] * qd[1] + qd[1] ** 2), -h * qd[0] ** 2], dtype=np.float64
    )
    gravity = np.array(
        [
            (m1 * lc1 + m2 * l1) * g * np.cos(q[0]) + m2 * lc2 * g * np.cos(q[0] + q[1]),
            m2 * lc2 * g * np.cos(q[0] + q[1]),
        ],
        dtype=np.float64,
    )
    damping = np.array([b1 * qd[0], b2 * qd[1]], dtype=np.float64)
    rhs = np.asarray(u, dtype=np.float64) - coriolis - gravity - damping
    solution: np.ndarray = np.linalg.solve(mass, rhs)
    return solution
