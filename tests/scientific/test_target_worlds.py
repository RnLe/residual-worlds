"""Target-world composition: component identities, toggles, scaling, energy."""

import numpy as np
import pytest
import torch

from residual_worlds.config import load_contract
from residual_worlds.paths import repository_root
from residual_worlds.physics import kinematics, nominal
from residual_worlds.physics.integrators import rk4_transition
from residual_worlds.physics.target import (
    ResolvedWorld,
    applied_torque,
    elastic_potential,
    elastic_torque,
    friction_torque,
    payload_mass_matrix,
    resolve_world,
    target_acceleration,
)

pytestmark = pytest.mark.scientific

CONTRACT = load_contract(repository_root() / "configs" / "experiment_contract.yaml")
ARM = CONTRACT.arm


def _random_states(count: int, seed: int = 23) -> torch.Tensor:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    q = np.stack(
        [rng.uniform(ARM.q_min_rad[j] + 0.05, ARM.q_max_rad[j] - 0.05, count) for j in range(2)],
        axis=-1,
    )
    qd = rng.uniform(-5.0, 5.0, size=(count, 2))
    return torch.from_numpy(np.concatenate([q, qd], axis=-1))


def _random_actions(count: int, seed: int = 29) -> torch.Tensor:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    return torch.from_numpy(rng.uniform(-4.0, 4.0, size=(count, 2)))


def test_nominal_sanity_equals_nominal() -> None:
    world = resolve_world(CONTRACT, "nominal_sanity")
    states, actions = _random_states(128), _random_actions(128)
    target = target_acceleration(states, actions, world, ARM)
    reference = nominal.state_acceleration(states, actions, ARM)
    torch.testing.assert_close(target, reference, atol=0.0, rtol=0.0)


def test_payload_mass_matrix_jacobian_identity() -> None:
    # A point mass at the end effector contributes exactly m_p J^T J.
    world = resolve_world(CONTRACT, "payload_standard")
    assert world.payload_kg is not None
    states = _random_states(64)
    q = states[:, :2]
    jacobian = kinematics.end_effector_jacobian(q, ARM)
    identity = world.payload_kg * torch.einsum("...ki,...kj->...ij", jacobian, jacobian)
    torch.testing.assert_close(
        payload_mass_matrix(q, world.payload_kg, ARM), identity, atol=1e-12, rtol=1e-12
    )


def test_components_toggle_independently() -> None:
    states, actions = _random_states(64), _random_actions(64)
    nominal_accel = nominal.state_acceleration(states, actions, ARM)
    for world_id, expect_difference in (
        ("payload_standard", True),
        ("friction_standard", True),
        ("actuator_standard", True),
    ):
        world = resolve_world(CONTRACT, world_id)
        accel = target_acceleration(states, actions, world, ARM)
        assert torch.isfinite(accel).all()
        differs = not torch.allclose(accel, nominal_accel, atol=1e-9)
        assert differs == expect_difference, world_id


def test_friction_with_zero_stribeck_terms_equals_nominal_damping() -> None:
    # The friction law replaces B0 qd; with Coulomb and peak levels at
    # zero it must reduce to exactly the nominal viscous torque.
    base = resolve_world(CONTRACT, "friction_standard")
    assert base.friction is not None
    from dataclasses import replace

    degenerate = replace(base.friction, coulomb_nm=(0.0, 0.0), low_speed_peak_nm=(0.0, 0.0))
    qd = _random_states(64)[:, 2:]
    torch.testing.assert_close(
        friction_torque(qd, degenerate), nominal.damping_torque(qd, ARM)
    )


def test_friction_opposes_velocity() -> None:
    world = resolve_world(CONTRACT, "friction_standard")
    assert world.friction is not None
    qd = _random_states(256)[:, 2:]
    torque = friction_torque(qd, world.friction)
    # Dissipative sign convention: friction torque has the sign of qd
    # (it is subtracted on the left / opposes motion), and vanishes at rest.
    assert torch.all(torque * torch.sign(qd) >= 0.0)
    zero = friction_torque(torch.zeros(2, dtype=torch.float64), world.friction)
    torch.testing.assert_close(zero, torch.zeros(2, dtype=torch.float64))


def test_actuator_deadzone_and_continuity() -> None:
    world = resolve_world(CONTRACT, "actuator_standard")
    assert world.actuator is not None
    deadzone = world.actuator.deadzone_nm
    inside = torch.tensor([deadzone[0] * 0.5, -deadzone[1] * 0.9], dtype=torch.float64)
    torch.testing.assert_close(
        applied_torque(inside, world.actuator, ARM), torch.zeros(2, dtype=torch.float64)
    )
    # Continuity across each dead-zone corner.
    for joint in range(2):
        below = torch.zeros(2, dtype=torch.float64)
        above = torch.zeros(2, dtype=torch.float64)
        below[joint] = deadzone[joint] - 1e-9
        above[joint] = deadzone[joint] + 1e-9
        difference = applied_torque(above, world.actuator, ARM) - applied_torque(
            below, world.actuator, ARM
        )
        assert float(torch.abs(difference).max()) < 1e-6
    # Clipping to the physical limit after gain.
    huge = torch.tensor([100.0, -100.0], dtype=torch.float64)
    clipped = applied_torque(huge, world.actuator, ARM)
    limit = torch.tensor(ARM.torque_limit_nm, dtype=torch.float64)
    assert torch.all(torch.abs(clipped) <= limit + 1e-12)


def test_elastic_torque_is_negative_potential_gradient() -> None:
    k_c = 0.18
    q = _random_states(64)[:, :2].clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(elastic_potential(q, k_c).sum(), q)
    torch.testing.assert_close(elastic_torque(q.detach(), k_c), -gradient, atol=1e-12, rtol=1e-12)


def test_magnitude_scaling_rule() -> None:
    standard = resolve_world(CONTRACT, "composite_standard")
    low = resolve_world(CONTRACT, "composite_low")
    assert standard.payload_kg is not None and low.payload_kg is not None
    assert standard.friction is not None and low.friction is not None
    assert standard.actuator is not None and low.actuator is not None
    scale = 0.6
    assert low.payload_kg == pytest.approx(scale * standard.payload_kg)
    for joint in range(2):
        assert low.friction.coulomb_nm[joint] == pytest.approx(
            scale * standard.friction.coulomb_nm[joint]
        )
        assert low.friction.low_speed_peak_nm[joint] == pytest.approx(
            scale * standard.friction.low_speed_peak_nm[joint]
        )
        assert low.actuator.deadzone_nm[joint] == pytest.approx(
            scale * standard.actuator.deadzone_nm[joint]
        )
        assert low.actuator.gain[joint] == pytest.approx(
            1.0 + scale * (standard.actuator.gain[joint] - 1.0)
        )
        # Never scaled: these define the shape, not the magnitude.
        assert low.friction.viscous_nm_s_rad[joint] == standard.friction.viscous_nm_s_rad[joint]
        assert (
            low.friction.stribeck_velocity_rad_s[joint]
            == standard.friction.stribeck_velocity_rad_s[joint]
        )
        assert (
            low.friction.smoothing_velocity_rad_s[joint]
            == standard.friction.smoothing_velocity_rad_s[joint]
        )


def test_elastic_unseen_extends_composite() -> None:
    composite = resolve_world(CONTRACT, "composite_standard")
    elastic = resolve_world(CONTRACT, "elastic_unseen")
    assert elastic.elastic_coupling_nm == pytest.approx(0.18)
    assert elastic.payload_kg == composite.payload_kg
    assert elastic.friction == composite.friction
    assert elastic.actuator == composite.actuator
    states, actions = _random_states(32), _random_actions(32)
    difference = target_acceleration(states, actions, elastic, ARM) - target_acceleration(
        states, actions, composite, ARM
    )
    assert float(torch.abs(difference).max()) > 0.0


def test_elastic_energy_dissipates_without_input() -> None:
    # Nominal damping + elastic coupling, zero torque: T + V + V_el is
    # non-increasing (the elastic term is conservative, damping is not).
    k_c = 0.18
    world = ResolvedWorld(
        world_id="elastic_only_test",
        payload_kg=None,
        friction=None,
        actuator=None,
        elastic_coupling_nm=k_c,
    )
    state = torch.tensor([1.0, -0.8, 1.2, -0.6], dtype=torch.float64)
    action = torch.zeros(2, dtype=torch.float64)

    def accel(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return target_acceleration(s, a, world, ARM)

    def total_energy(s: torch.Tensor) -> float:
        return float(
            nominal.kinetic_energy(s[:2], s[2:], ARM)
            + nominal.potential_energy(s[:2], ARM)
            + elastic_potential(s[:2], k_c)
        )

    energies = [total_energy(state)]
    current = state.clone()
    for _ in range(80):
        current = rk4_transition(accel, current, action, 0.05, 8)
        energies.append(total_energy(current))
    increases = [b - a for a, b in zip(energies[:-1], energies[1:], strict=True)]
    assert max(increases) < 1e-9


def test_composite_uses_actuator_on_command_not_on_elastic() -> None:
    # The elastic torque enters the right-hand side directly; only the
    # commanded torque passes the actuator transform. With u inside the
    # dead zone, the composed acceleration must match the same world
    # with the actuator applied to zero torque.
    elastic = resolve_world(CONTRACT, "elastic_unseen")
    assert elastic.actuator is not None
    tiny = torch.tensor(
        [elastic.actuator.deadzone_nm[0] * 0.5, -elastic.actuator.deadzone_nm[1] * 0.5],
        dtype=torch.float64,
    )
    states = _random_states(16)
    with_tiny = target_acceleration(states, tiny.expand(16, 2), elastic, ARM)
    with_zero = target_acceleration(states, torch.zeros(16, 2, dtype=torch.float64), elastic, ARM)
    torch.testing.assert_close(with_tiny, with_zero)
