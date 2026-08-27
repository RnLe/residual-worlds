"""Target-world dynamics: nominal physics plus controlled, hidden mismatch.

Every composed world obeys one canonical equation,

    M_w(q) qdd + c_w(q, qd) + g_w(q) + tau_fric(qd)
        = tau_act(u) + tau_elastic(q),

with the following component semantics (each reduces exactly to the
nominal term when disabled):

* payload: a point mass m_p at the end effector adds M_p, c_p, g_p on
  the left (M_p equals m_p J_e^T J_e, tested against that identity);
* nonlinear friction: a smooth Stribeck-like law REPLACES the nominal
  viscous term B0 qd -- it contains its own viscous coefficient, so
  nominal damping is never double-counted;
* actuator: commanded torque is transformed by gain and dead zone,
  then componentwise clipped to the physical limit;
* elastic coupling: the conservative torque -grad V of
  V = k_c (1 - cos(q1 - q2)), a synthetic joint coupling used as the
  held-out mechanism.

Magnitude scaling s (for the transfer worlds) multiplies m_p, the
Coulomb and low-speed-peak levels, and the dead zone; the gain maps to
1 + s (gain - 1). Viscous, Stribeck, and smoothing scales never scale,
so s = 0 recovers the nominal world exactly.

This module is importable only by the simulator/evaluation side.
Learned-model, training, and planning code must never import it; a
static boundary test enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from residual_worlds.config import ActuatorSpec, ExperimentContract, FrictionSpec, WorldSpec
from residual_worlds.physics import nominal
from residual_worlds.physics.components import (
    payload_coriolis,
    payload_gravity,
    payload_mass_matrix,
)
from residual_worlds.types import ArmParameters


@dataclass(frozen=True)
class ResolvedWorld:
    """A fully resolved target world (base inheritance and scaling applied)."""

    world_id: str
    payload_kg: float | None
    friction: FrictionSpec | None
    actuator: ActuatorSpec | None
    elastic_coupling_nm: float | None


def _scale_spec(spec: WorldSpec, base: ResolvedWorld, scale: float) -> ResolvedWorld:
    payload = None if base.payload_kg is None else scale * base.payload_kg
    friction = base.friction
    if friction is not None:
        friction = replace(
            friction,
            coulomb_nm=(scale * friction.coulomb_nm[0], scale * friction.coulomb_nm[1]),
            low_speed_peak_nm=(
                scale * friction.low_speed_peak_nm[0],
                scale * friction.low_speed_peak_nm[1],
            ),
        )
    actuator = base.actuator
    if actuator is not None:
        actuator = ActuatorSpec(
            gain=(
                1.0 + scale * (actuator.gain[0] - 1.0),
                1.0 + scale * (actuator.gain[1] - 1.0),
            ),
            deadzone_nm=(scale * actuator.deadzone_nm[0], scale * actuator.deadzone_nm[1]),
        )
    return ResolvedWorld(
        world_id=spec.world_id,
        payload_kg=payload,
        friction=friction,
        actuator=actuator,
        elastic_coupling_nm=base.elastic_coupling_nm,
    )


def resolve_world(contract: ExperimentContract, world_id: str) -> ResolvedWorld:
    """Expand base-world inheritance and magnitude scaling into flat parameters."""
    spec = contract.worlds.get(world_id)
    if spec is None:
        raise KeyError(f"unknown world {world_id!r}")
    if spec.base_world is None:
        return ResolvedWorld(
            world_id=world_id,
            payload_kg=spec.payload_kg if "payload" in spec.components else None,
            friction=spec.friction if "nonlinear_friction" in spec.components else None,
            actuator=spec.actuator if "actuator" in spec.components else None,
            elastic_coupling_nm=None,
        )
    base = resolve_world(contract, spec.base_world)
    resolved = base
    if spec.magnitude_scale is not None:
        resolved = _scale_spec(spec, base, spec.magnitude_scale)
    else:
        resolved = replace(base, world_id=world_id)
    if "elastic_coupling" in spec.components:
        resolved = replace(
            resolved, world_id=world_id, elastic_coupling_nm=spec.elastic_coupling_nm
        )
    else:
        resolved = replace(resolved, world_id=world_id)
    return resolved


# ---------------------------------------------------------------------------
# Component terms


def friction_torque(qd: torch.Tensor, friction: FrictionSpec) -> torch.Tensor:
    """Smooth Stribeck-like joint friction (replaces nominal viscous damping).

    Per joint: b qd + [f_c + (f_s - f_c) exp(-(qd / v_s)^2)] tanh(qd / eps).
    At exactly zero velocity this produces zero force -- it is a smooth
    surrogate, not set-valued static friction.
    """
    b = torch.as_tensor(friction.viscous_nm_s_rad, dtype=qd.dtype, device=qd.device)
    fc = torch.as_tensor(friction.coulomb_nm, dtype=qd.dtype, device=qd.device)
    fs = torch.as_tensor(friction.low_speed_peak_nm, dtype=qd.dtype, device=qd.device)
    vs = torch.as_tensor(friction.stribeck_velocity_rad_s, dtype=qd.dtype, device=qd.device)
    eps = torch.as_tensor(friction.smoothing_velocity_rad_s, dtype=qd.dtype, device=qd.device)
    stribeck = fc + (fs - fc) * torch.exp(-((qd / vs) ** 2))
    return b * qd + stribeck * torch.tanh(qd / eps)


def applied_torque(
    u: torch.Tensor, actuator: ActuatorSpec | None, arm: ArmParameters
) -> torch.Tensor:
    """Hidden command-to-applied-torque map: gain, dead zone, clip."""
    if actuator is None:
        return u
    gain = torch.as_tensor(actuator.gain, dtype=u.dtype, device=u.device)
    deadzone = torch.as_tensor(actuator.deadzone_nm, dtype=u.dtype, device=u.device)
    limit = torch.as_tensor(arm.torque_limit_nm, dtype=u.dtype, device=u.device)
    magnitude = torch.clamp(torch.abs(u) - deadzone, min=0.0)
    return torch.clamp(gain * torch.sign(u) * magnitude, min=-limit, max=limit)


def elastic_torque(q: torch.Tensor, k_c: float) -> torch.Tensor:
    """Conservative coupling torque -grad V for V = k_c (1 - cos(q1 - q2))."""
    s = torch.sin(q[..., 0] - q[..., 1])
    return torch.stack((-k_c * s, k_c * s), dim=-1)


def elastic_potential(q: torch.Tensor, k_c: float) -> torch.Tensor:
    return k_c * (1.0 - torch.cos(q[..., 0] - q[..., 1]))


# ---------------------------------------------------------------------------
# Composed target acceleration


def target_acceleration(
    state: torch.Tensor, action: torch.Tensor, world: ResolvedWorld, arm: ArmParameters
) -> torch.Tensor:
    """Joint acceleration of the composed target world.

    ``action`` is the commanded torque (already clipped to the command
    bound by the caller); the hidden actuator transform happens here.
    """
    q, qd = state[..., :2], state[..., 2:]
    mass = nominal.mass_matrix(q, arm)
    coriolis = nominal.coriolis_vector(q, qd, arm)
    gravity = nominal.gravity_vector(q, arm)
    if world.payload_kg is not None:
        mass = mass + payload_mass_matrix(q, world.payload_kg, arm)
        coriolis = coriolis + payload_coriolis(q, qd, world.payload_kg, arm)
        gravity = gravity + payload_gravity(q, world.payload_kg, arm)
    if world.friction is not None:
        friction = friction_torque(qd, world.friction)
    else:
        friction = nominal.damping_torque(qd, arm)
    right = applied_torque(action, world.actuator, arm)
    if world.elastic_coupling_nm is not None:
        right = right + elastic_torque(q, world.elastic_coupling_nm)
    rhs = right - coriolis - gravity - friction
    solution: torch.Tensor = torch.linalg.solve(mass, rhs.unsqueeze(-1))
    return solution.squeeze(-1)
