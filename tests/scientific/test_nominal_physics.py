"""Analytic and numerical checks of kinematics, nominal dynamics, and RK4."""

import math

import numpy as np
import pytest
import torch

from residual_worlds.config import load_contract
from residual_worlds.paths import repository_root
from residual_worlds.physics import kinematics, nominal
from residual_worlds.physics.integrators import rk4_substep_endpoints, rk4_transition

pytestmark = pytest.mark.scientific

CONTRACT = load_contract(repository_root() / "configs" / "experiment_contract.yaml")
ARM = CONTRACT.arm


def _random_states(count: int, seed: int = 7) -> torch.Tensor:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    q = np.stack(
        [rng.uniform(ARM.q_min_rad[j] + 0.05, ARM.q_max_rad[j] - 0.05, count) for j in range(2)],
        axis=-1,
    )
    qd = rng.uniform(-6.0, 6.0, size=(count, 2))
    return torch.from_numpy(np.concatenate([q, qd], axis=-1))


# ---------------------------------------------------------------------------
# Kinematics


def test_forward_kinematics_hand_poses() -> None:
    l1, l2 = ARM.link_lengths_m
    # Fully horizontal arm.
    q = torch.tensor([0.0, 0.0], dtype=torch.float64)
    ee = kinematics.end_effector_position(q, ARM)
    torch.testing.assert_close(ee, torch.tensor([l1 + l2, 0.0], dtype=torch.float64))
    # Vertical shoulder, right-angle elbow bending back to horizontal.
    q = torch.tensor([math.pi / 2, -math.pi / 2], dtype=torch.float64)
    ee = kinematics.end_effector_position(q, ARM)
    torch.testing.assert_close(ee, torch.tensor([l2, l1], dtype=torch.float64))
    # Folded arm: elbow at pi points link 2 back onto link 1.
    q = torch.tensor([0.0, math.pi], dtype=torch.float64)
    ee = kinematics.end_effector_position(q, ARM)
    torch.testing.assert_close(
        ee, torch.tensor([l1 - l2, 0.0], dtype=torch.float64), atol=1e-12, rtol=0.0
    )


def test_jacobian_matches_finite_differences() -> None:
    states = _random_states(64)
    q = states[:, :2]
    analytic = kinematics.end_effector_jacobian(q, ARM)
    eps = 1e-7
    for joint in range(2):
        offset = torch.zeros_like(q)
        offset[:, joint] = eps
        numeric = (
            kinematics.end_effector_position(q + offset, ARM)
            - kinematics.end_effector_position(q - offset, ARM)
        ) / (2 * eps)
        torch.testing.assert_close(analytic[:, :, joint], numeric, atol=1e-6, rtol=1e-6)


def test_end_effector_velocity_matches_position_derivative() -> None:
    states = _random_states(16)
    q, qd = states[:, :2], states[:, 2:]
    velocity = kinematics.end_effector_velocity(q, qd, ARM)
    dt = 1e-7
    numeric = (
        kinematics.end_effector_position(q + dt * qd, ARM)
        - kinematics.end_effector_position(q - dt * qd, ARM)
    ) / (2 * dt)
    torch.testing.assert_close(velocity, numeric, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Nominal dynamics


def test_mass_matrix_symmetric_positive_definite() -> None:
    states = _random_states(256)
    m = nominal.mass_matrix(states[:, :2], ARM)
    torch.testing.assert_close(m, m.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(m)
    assert float(eigenvalues.min()) > 1e-4


def test_gravity_is_gradient_of_potential() -> None:
    states = _random_states(128)
    q = states[:, :2].clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(nominal.potential_energy(q, ARM).sum(), q)
    analytic = nominal.gravity_vector(q.detach(), ARM)
    torch.testing.assert_close(gradient, analytic, atol=1e-10, rtol=1e-10)


def test_batched_acceleration_matches_scalar_reference() -> None:
    states = _random_states(64)
    rng = np.random.Generator(np.random.PCG64DXSM(11))
    actions = torch.from_numpy(rng.uniform(-4.0, 4.0, size=(64, 2)))
    batched = nominal.state_acceleration(states, actions, ARM).numpy()
    for index in range(64):
        reference = nominal.acceleration_reference_numpy(
            states[index, :2].numpy(), states[index, 2:].numpy(), actions[index].numpy(), ARM
        )
        np.testing.assert_allclose(batched[index], reference, atol=1e-11)


def test_power_balance_identity() -> None:
    # d(T + V)/dt must equal qd . (u - B qd): actuation power minus
    # viscous dissipation. Verified by a tiny centered difference on a
    # very short exact-dynamics step.
    states = _random_states(32)
    rng = np.random.Generator(np.random.PCG64DXSM(13))
    actions = torch.from_numpy(rng.uniform(-4.0, 4.0, size=(32, 2)))

    def accel(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return nominal.state_acceleration(s, a, ARM)

    dt = 1e-6
    forward = rk4_transition(accel, states, actions, dt, 1)
    backward = rk4_transition(accel, states, actions, -dt, 1)

    def total_energy(s: torch.Tensor) -> torch.Tensor:
        return nominal.kinetic_energy(s[:, :2], s[:, 2:], ARM) + nominal.potential_energy(
            s[:, :2], ARM
        )

    energy_rate = (total_energy(forward) - total_energy(backward)) / (2 * dt)
    qd = states[:, 2:]
    expected = torch.einsum(
        "...i,...i->...", qd, actions - nominal.damping_torque(qd, ARM)
    )
    torch.testing.assert_close(energy_rate, expected, atol=1e-4, rtol=1e-4)


def test_kinetic_energy_rate_consistency() -> None:
    # Energy surrogate for the skew-symmetry property of dM - 2C: along
    # the dynamics, d/dt (qd^T M qd / 2) computed by the chain rule
    # (which needs dM/dt) must equal the mechanical power
    # qd^T (u - g - B qd). This holds only if the implemented Coriolis
    # vector is consistent with the configuration dependence of M.
    states = _random_states(32)
    q, qd = states[:, :2], states[:, 2:]
    rng = np.random.Generator(np.random.PCG64DXSM(17))
    u = torch.from_numpy(rng.uniform(-4.0, 4.0, size=(32, 2)))
    qdd = nominal.acceleration(q, qd, u, ARM)

    # Kinetic-energy rate two ways: chain rule with dM/dq2, and power form.
    eps = 1e-7
    offset = torch.zeros_like(q)
    offset[:, 1] = eps
    dm_dq2 = (nominal.mass_matrix(q + offset, ARM) - nominal.mass_matrix(q - offset, ARM)) / (
        2 * eps
    )
    m_dot = dm_dq2 * qd[:, 1].reshape(-1, 1, 1)
    kinetic_rate_chain = torch.einsum("...i,...ij,...j->...", qd, nominal.mass_matrix(q, ARM), qdd)
    kinetic_rate_chain = kinetic_rate_chain + 0.5 * torch.einsum(
        "...i,...ij,...j->...", qd, m_dot, qd
    )
    power_form = torch.einsum(
        "...i,...i->...",
        qd,
        u - nominal.gravity_vector(q, ARM) - nominal.damping_torque(qd, ARM),
    )
    torch.testing.assert_close(kinetic_rate_chain, power_form, atol=1e-5, rtol=1e-5)


def test_extreme_inputs_stay_finite() -> None:
    states = _random_states(64)
    limit = torch.tensor(ARM.torque_limit_nm, dtype=torch.float64)
    for action in (limit, -limit, torch.zeros(2, dtype=torch.float64)):
        qdd = nominal.state_acceleration(states, action.expand(64, 2), ARM)
        assert torch.isfinite(qdd).all()


# ---------------------------------------------------------------------------
# Integrator


def test_rk4_exact_on_harmonic_oscillator_order() -> None:
    # qdd = -omega^2 q has known solution; global RK4 error ~ h^4.
    omega = 3.0

    def accel(state: torch.Tensor, _action: torch.Tensor) -> torch.Tensor:
        return -(omega**2) * state[..., :2]

    initial = torch.tensor([0.3, -0.2, 0.0, 0.0], dtype=torch.float64)
    action = torch.zeros(2, dtype=torch.float64)
    duration = 1.0

    def final_error(substeps_per_call: int) -> float:
        state = initial.clone()
        steps = 20
        for _ in range(steps):
            state = rk4_transition(accel, state, action, duration / steps, substeps_per_call)
        exact_q = initial[:2] * math.cos(omega * duration)
        return float(torch.max(torch.abs(state[:2] - exact_q)))

    e1, e2 = final_error(1), final_error(2)
    assert e1 / e2 > 12.0  # fourth order: ratio ~ 16


def test_rk4_zero_duration_is_identity() -> None:
    def accel(state: torch.Tensor, _action: torch.Tensor) -> torch.Tensor:
        return nominal.state_acceleration(state, _action, ARM)

    state = _random_states(4)
    action = torch.zeros(4, 2, dtype=torch.float64)
    stepped = rk4_transition(accel, state, action, 0.0, 1)
    torch.testing.assert_close(stepped, state)


def test_rk4_substep_endpoints_shape_and_consistency() -> None:
    def accel(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return nominal.state_acceleration(state, action, ARM)

    states = _random_states(8)
    actions = torch.zeros(8, 2, dtype=torch.float64)
    endpoints = rk4_substep_endpoints(accel, states, actions, 0.05, 5)
    assert endpoints.shape == (8, 6, 4)
    torch.testing.assert_close(endpoints[:, 0], states)
    # Last endpoint equals the transition function's answer.
    torch.testing.assert_close(
        endpoints[:, -1], rk4_transition(accel, states, actions, 0.05, 5)
    )
    # Two substeps of h/2 equal one call with substeps=2.
    two = rk4_transition(accel, states, actions, 0.05, 2)
    half = rk4_transition(accel, states, actions, 0.025, 1)
    half = rk4_transition(accel, half, actions, 0.025, 1)
    torch.testing.assert_close(two, half)
