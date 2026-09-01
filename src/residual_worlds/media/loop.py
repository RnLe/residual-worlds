"""One periodic episode of the arm in a target world, plus what the nominal
model imagined along the way.

The torques come from a computed-torque tracker built on the exact
target dynamics (the study's exact-dynamics reference), so the
underpowered arm follows a smooth joint-space reference without hitting
its torque or speed limits. From every recorded state the nominal model
is then rolled forward for ``ghost_steps`` under those same torques, so
frame k can show what the model imagined ``ghost_steps`` ago next to
what happened. Who chose the torques does not matter for that picture;
only that model and world receive the same ones. The site mirrors this
loop in TypeScript; ``golden.py`` pins both to the same numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from residual_worlds.config import ExperimentContract
from residual_worlds.physics import nominal
from residual_worlds.physics.components import (
    payload_coriolis,
    payload_gravity,
    payload_mass_matrix,
)
from residual_worlds.physics.integrators import rk4_transition
from residual_worlds.physics.target import (
    ResolvedWorld,
    elastic_torque,
    friction_torque,
    resolve_world,
    target_acceleration,
)
from residual_worlds.types import ArmParameters


@dataclass(frozen=True)
class Schedule:
    """Reference motion and tracker gains; every constant is shared with the site."""

    period_s: float = 7.0
    # The arm is weak against gravity, so the reference keeps its centre of
    # mass over the base: the forearm folds against the shoulder lean.
    q_center_rad: tuple[float, float] = (1.856, -1.0)
    amplitude_rad: tuple[float, float] = (0.3, 0.75)
    phase_rad: tuple[float, float] = (0.0, math.pi)
    harmonics: tuple[int, int] = (1, 1)
    kp: float = 25.0  # per unit inertia
    kd: float = 7.0
    warmup_periods: int = 2
    ghost_steps: int = 6


SCHEDULE = Schedule()


@dataclass(frozen=True)
class Loop:
    """Recorded period: arrays indexed by control step."""

    dt_s: float
    states: np.ndarray  # [N, 4]
    actions: np.ndarray  # [N, 2]
    nominal_acc: np.ndarray  # [N, 2]
    target_acc: np.ndarray  # [N, 2]
    ghosts: np.ndarray  # [N, ghost_steps + 1, 4], nominal rollout launched at step k

    @property
    def residual(self) -> np.ndarray:
        return np.asarray(self.target_acc - self.nominal_acc)

    @property
    def frames(self) -> int:
        return int(self.states.shape[0])


def reference(
    t: float, schedule: Schedule = SCHEDULE
) -> tuple[list[float], list[float], list[float]]:
    """Joint reference with its first two time derivatives at time ``t``."""
    q: list[float] = []
    qd: list[float] = []
    qdd: list[float] = []
    for j in range(2):
        omega = 2.0 * math.pi * schedule.harmonics[j] / schedule.period_s
        arg = omega * t + schedule.phase_rad[j]
        amplitude = schedule.amplitude_rad[j]
        q.append(schedule.q_center_rad[j] + amplitude * math.sin(arg))
        qd.append(amplitude * omega * math.cos(arg))
        qdd.append(-amplitude * omega * omega * math.sin(arg))
    return q, qd, qdd


def tracking_torque(
    state: torch.Tensor,
    t: float,
    arm: ArmParameters,
    world: ResolvedWorld,
    schedule: Schedule = SCHEDULE,
) -> torch.Tensor:
    """Computed torque on the exact target dynamics, clipped to the command bound.

    The commanded acceleration is the reference acceleration plus PD
    feedback; the world's inertia, Coriolis, gravity, and friction terms
    turn it into an applied torque, and the actuator map (gain, dead
    zone) is inverted so the command produces that torque.
    """
    q_ref, qd_ref, qdd_ref = reference(t, schedule)
    q, qd = state[:2], state[2:]
    dtype = state.dtype
    ref_q = torch.tensor(q_ref, dtype=dtype)
    ref_qd = torch.tensor(qd_ref, dtype=dtype)
    ref_qdd = torch.tensor(qdd_ref, dtype=dtype)
    command = ref_qdd + schedule.kp * (ref_q - q) + schedule.kd * (ref_qd - qd)

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
    applied = torch.einsum("ij,j->i", mass, command) + coriolis + gravity + friction
    if world.elastic_coupling_nm is not None:
        applied = applied - elastic_torque(q, world.elastic_coupling_nm)

    if world.actuator is not None:
        gain = torch.tensor(world.actuator.gain, dtype=dtype)
        deadzone = torch.tensor(world.actuator.deadzone_nm, dtype=dtype)
        u = torch.sign(applied) * (torch.abs(applied) / gain + deadzone)
    else:
        u = applied
    limit = torch.tensor(arm.torque_limit_nm, dtype=dtype)
    return torch.clamp(u, min=-limit, max=limit)


def simulate_loop(
    contract: ExperimentContract,
    world_id: str = "composite_standard",
    schedule: Schedule = SCHEDULE,
) -> Loop:
    arm = contract.arm
    world = resolve_world(contract, world_id)
    dt = contract.numerics.control_dt_s
    substeps = contract.numerics.substeps_per_control_step
    steps = int(round(schedule.period_s / dt))

    def true_acc(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return target_acceleration(s, a, world, arm)

    def nominal_acc(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return nominal.state_acceleration(s, a, arm)

    q0, qd0, _ = reference(0.0, schedule)
    state = torch.tensor([q0[0], q0[1], qd0[0], qd0[1]], dtype=torch.float64)

    # Warm up so the recorded period starts on the tracker's steady cycle.
    for k in range(schedule.warmup_periods * steps):
        action = tracking_torque(state, k * dt, arm, world, schedule)
        state = rk4_transition(true_acc, state, action, dt, substeps)

    states = torch.empty((steps, 4), dtype=torch.float64)
    actions = torch.empty((steps, 2), dtype=torch.float64)
    t0 = schedule.warmup_periods * schedule.period_s
    for k in range(steps):
        states[k] = state
        actions[k] = tracking_torque(state, t0 + k * dt, arm, world, schedule)
        state = rk4_transition(true_acc, state, actions[k], dt, substeps)

    nominal_accs = nominal_acc(states, actions)
    target_accs = true_acc(states, actions)

    ghosts = torch.empty((steps, schedule.ghost_steps + 1, 4), dtype=torch.float64)
    for k in range(steps):
        ghost = states[k]
        ghosts[k, 0] = ghost
        for i in range(schedule.ghost_steps):
            ghost = rk4_transition(nominal_acc, ghost, actions[(k + i) % steps], dt, substeps)
            ghosts[k, i + 1] = ghost

    return Loop(
        dt_s=dt,
        states=states.numpy(),
        actions=actions.numpy(),
        nominal_acc=nominal_accs.numpy(),
        target_acc=target_accs.numpy(),
        ghosts=ghosts.numpy(),
    )
