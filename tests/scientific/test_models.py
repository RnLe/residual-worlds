"""Model conditions: capacity matching, zero-residual identity, fitted recovery."""

import numpy as np
import pytest
import torch

from residual_worlds.config import load_contract
from residual_worlds.models.base import Stepper
from residual_worlds.models.black_box import BlackBoxModel
from residual_worlds.models.fitted_physics import (
    FittedPhysicsModel,
    fit_fitted_physics,
    fitted_acceleration,
    nominal_theta,
)
from residual_worlds.models.normalization import PhysicalScales, features
from residual_worlds.models.residual import ResidualModel
from residual_worlds.paths import repository_root
from residual_worlds.physics.nominal import state_acceleration
from residual_worlds.seeds import torch_generator

pytestmark = pytest.mark.scientific

CONTRACT = load_contract(repository_root() / "configs" / "experiment_contract.yaml")
ARM = CONTRACT.arm
SCALES = PhysicalScales.from_contract(CONTRACT)
ROOT = CONTRACT.numerics.root_seed


def _random_batch(count: int, seed: int = 41) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    q = np.stack(
        [rng.uniform(ARM.q_min_rad[j] + 0.1, ARM.q_max_rad[j] - 0.1, count) for j in range(2)],
        axis=-1,
    )
    qd = rng.uniform(-3.0, 3.0, size=(count, 2))
    u = rng.uniform(-3.5, 3.5, size=(count, 2))
    states = torch.from_numpy(np.concatenate([q, qd], axis=-1))
    return states, torch.from_numpy(u)


def _blackbox(member: int = 0) -> BlackBoxModel:
    generator = torch_generator(ROOT, "model_init", "blackbox", 0, member)
    return BlackBoxModel(CONTRACT.models.neural_common, SCALES, generator).double()


def _residual(member: int = 0) -> ResidualModel:
    generator = torch_generator(ROOT, "model_init", "residual", 0, member)
    return ResidualModel(CONTRACT.models.neural_common, SCALES, ARM, generator).double()


def test_parameter_counts_match_exactly() -> None:
    blackbox = _blackbox()
    residual = _residual()
    count_b = sum(p.numel() for p in blackbox.parameters())
    count_r = sum(p.numel() for p in residual.parameters())
    assert count_b == count_r
    # 8 -> 128 -> 128 -> 128 -> 2 with biases.
    expected = (8 * 128 + 128) + 2 * (128 * 128 + 128) + (128 * 2 + 2)
    assert count_b == expected


def test_zero_initialized_residual_is_exactly_nominal() -> None:
    residual = _residual()
    states, actions = _random_batch(128)
    predicted = residual.acceleration(states, actions)
    reference = state_acceleration(states, actions, ARM)
    torch.testing.assert_close(predicted, reference, atol=0.0, rtol=0.0)
    correction = residual.correction(states, actions).detach()
    assert float(correction.abs().max()) == 0.0


def test_blackbox_initialization_is_not_nominal() -> None:
    blackbox = _blackbox()
    states, actions = _random_batch(32)
    predicted = blackbox.acceleration(states, actions)
    reference = state_acceleration(states, actions, ARM)
    assert not torch.allclose(predicted, reference, atol=1e-3)


def test_member_initializations_are_seeded_and_distinct() -> None:
    a1, a2 = _blackbox(member=0), _blackbox(member=0)
    b = _blackbox(member=1)
    states, actions = _random_batch(16)
    torch.testing.assert_close(a1.acceleration(states, actions), a2.acceleration(states, actions))
    assert not torch.allclose(a1.acceleration(states, actions), b.acceleration(states, actions))


def test_feature_vector_layout() -> None:
    states, actions = _random_batch(8)
    phi = features(states, actions, SCALES)
    assert phi.shape == (8, 8)
    torch.testing.assert_close(phi[:, 0], torch.sin(states[:, 0]))
    torch.testing.assert_close(phi[:, 3], torch.cos(states[:, 1]))
    torch.testing.assert_close(phi[:, 4], states[:, 2] / 4.0)
    torch.testing.assert_close(phi[:, 6], actions[:, 0] / 4.0)


def test_both_networks_can_fit_a_synthetic_acceleration() -> None:
    # A smooth synthetic acceleration field must be learnable by both
    # architectures (basic capacity sanity, not a benchmark).
    def target_field(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                2.0 * torch.sin(states[:, 0]) + 0.5 * actions[:, 1],
                -1.5 * states[:, 3] + actions[:, 0],
            ),
            dim=-1,
        )

    states, actions = _random_batch(512)
    targets = target_field(states, actions)
    for model in (_blackbox(), _residual()):
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(400):
            optimizer.zero_grad()
            if isinstance(model, ResidualModel):
                predicted = model.correction(states, actions)
            else:
                predicted = model.acceleration(states, actions)
            loss = torch.mean((predicted - targets) ** 2)
            loss.backward()
            optimizer.step()
        assert float(loss) < 0.05, type(model).__name__


def _make_loss(theta_star: tuple[float, ...], count: int = 512):
    stepper = Stepper.from_contract(CONTRACT)
    states, actions = _random_batch(count, seed=57)
    with torch.no_grad():
        truth = stepper.step(
            lambda s, a: fitted_acceleration(
                s, a, torch.tensor(theta_star, dtype=torch.float64), ARM
            ),
            states,
            actions,
        )

    def loss(theta: torch.Tensor) -> torch.Tensor:
        predicted = stepper.step(
            lambda s, a: fitted_acceleration(s, a, theta, ARM), states, actions
        )
        return torch.mean((predicted - truth) ** 2)

    return loss


def test_fitted_physics_recovers_interior_parameters() -> None:
    theta_star = (0.25, 0.09, 0.07, 0.86, 1.12)
    loss = _make_loss(theta_star)
    result = fit_fitted_physics(CONTRACT, loss, loss, "test_world", 2048, 0)
    np.testing.assert_allclose(result.theta, theta_star, atol=1e-4)
    assert result.boundary_hits == ()


def test_fitted_physics_attains_exact_boundary() -> None:
    # True payload at the upper bound: the bounded optimizer must land
    # exactly on the boundary, not hover inside it.
    theta_star = (0.8, 0.05, 0.05, 1.0, 1.0)
    loss = _make_loss(theta_star)
    result = fit_fitted_physics(CONTRACT, loss, loss, "test_world", 2048, 1)
    np.testing.assert_allclose(result.theta, theta_star, atol=1e-4)
    assert "payload_kg" in result.boundary_hits


def test_fitted_physics_on_nominal_data_recovers_nominal() -> None:
    theta_star = nominal_theta(ARM)
    loss = _make_loss(theta_star)
    result = fit_fitted_physics(CONTRACT, loss, loss, "test_world", 2048, 2)
    np.testing.assert_allclose(result.theta, theta_star, atol=1e-4)


def test_fitted_model_class_matches_functional_form() -> None:
    theta = (0.2, 0.06, 0.08, 0.9, 1.1)
    model = FittedPhysicsModel(ARM, theta)
    states, actions = _random_batch(64)
    torch.testing.assert_close(
        model.acceleration(states, actions),
        fitted_acceleration(states, actions, torch.tensor(theta, dtype=torch.float64), ARM),
    )
