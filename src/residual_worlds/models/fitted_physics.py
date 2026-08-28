"""Bounded fitted-physics baseline: five interpretable parameters.

The model family deliberately contains only a point payload, diagonal
viscous damping (REPLACING the nominal damping, not adding to it), and
per-joint actuator gain followed by a componentwise clip:

    (M0 + M_p) qdd + (c0 + c_p) + (g0 + g_p) + diag(b) qd
        = clip(diag(alpha) u, -tau_max, tau_max),

with theta = (m_p, b1, b2, alpha1, alpha2) inside frozen physical
bounds. It excludes Coulomb/Stribeck friction, dead zones, and elastic
coupling by design: it asks whether ordinary low-dimensional
recalibration explains the mismatch before any neural correction is
justified.

Fitting uses deterministic bounded L-BFGS-B (SciPy driving a PyTorch
loss/gradient) so exact boundary values remain attainable, with the
all-nominal boundary point plus seeded interior restarts. Selection is
by validation loss with restart-index tie-breaking.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

from residual_worlds.config import ExperimentContract, FittedPhysicsConfig
from residual_worlds.physics import nominal
from residual_worlds.physics.components import (
    payload_coriolis,
    payload_gravity,
    payload_mass_matrix,
)
from residual_worlds.seeds import numpy_generator
from residual_worlds.types import ArmParameters

PARAMETER_ORDER = ("payload_kg", "viscous_1", "viscous_2", "actuator_gain_1", "actuator_gain_2")


def fitted_acceleration(
    state: torch.Tensor, action: torch.Tensor, theta: torch.Tensor, arm: ArmParameters
) -> torch.Tensor:
    """Acceleration of the fitted family; differentiable in ``theta`` ``[5]``."""
    q, qd = state[..., :2], state[..., 2:]
    m_p, b1, b2, a1, a2 = theta[0], theta[1], theta[2], theta[3], theta[4]
    mass = nominal.mass_matrix(q, arm) + payload_mass_matrix(q, m_p, arm)
    coriolis = nominal.coriolis_vector(q, qd, arm) + payload_coriolis(q, qd, m_p, arm)
    gravity = nominal.gravity_vector(q, arm) + payload_gravity(q, m_p, arm)
    damping = torch.stack((b1 * qd[..., 0], b2 * qd[..., 1]), dim=-1)
    limit = torch.as_tensor(arm.torque_limit_nm, dtype=state.dtype, device=state.device)
    gains = torch.stack((a1, a2)).to(state.dtype)
    applied = torch.clamp(gains * action, min=-limit, max=limit)
    rhs = applied - coriolis - gravity - damping
    solution: torch.Tensor = torch.linalg.solve(mass, rhs.unsqueeze(-1))
    return solution.squeeze(-1)


class FittedPhysicsModel:
    """Planner-facing frozen fitted model (theta fixed after fitting)."""

    model_id = "fitted_physics"

    def __init__(self, arm: ArmParameters, theta: tuple[float, float, float, float, float]):
        self._arm = arm
        self.theta = theta

    def acceleration(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        theta = torch.tensor(self.theta, dtype=state.dtype, device=state.device)
        return fitted_acceleration(state, action, theta, self._arm)


@dataclass(frozen=True)
class FitRestart:
    restart_index: int
    initial_theta: tuple[float, ...]
    final_theta: tuple[float, ...]
    train_loss: float
    validation_loss: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class FitResult:
    theta: tuple[float, float, float, float, float]
    selected_restart: int
    restarts: tuple[FitRestart, ...]
    boundary_hits: tuple[str, ...]


def nominal_theta(arm: ArmParameters) -> tuple[float, float, float, float, float]:
    """The all-nominal point: no payload, nominal damping, unit gains."""
    return (0.0, arm.viscous_nm_s_rad[0], arm.viscous_nm_s_rad[1], 1.0, 1.0)


def fit_fitted_physics(
    contract: ExperimentContract,
    train_loss: Callable[[torch.Tensor], torch.Tensor],
    validation_loss: Callable[[torch.Tensor], torch.Tensor],
    world_id: str,
    budget: int,
    replicate: int,
) -> FitResult:
    """Deterministic bounded multi-restart fit.

    ``train_loss``/``validation_loss`` map a theta tensor ``[5]`` to a
    scalar; they encapsulate the shared one-step-plus-rollout transition
    objective so this optimizer knows nothing about datasets.
    """
    config: FittedPhysicsConfig = contract.models.fitted_physics
    if config.parameters != PARAMETER_ORDER:
        raise ValueError("fitted-physics parameter order is fixed by the model family")
    bounds = [config.bounds[name] for name in PARAMETER_ORDER]
    arm = contract.arm

    initials: list[tuple[float, ...]] = [nominal_theta(arm)]
    rng = numpy_generator(
        contract.numerics.root_seed, "fitted_restart", world_id, budget, replicate
    )
    for _ in range(config.deterministic_restarts - 1):
        initials.append(tuple(float(rng.uniform(low, high)) for low, high in bounds))

    def objective(theta_array: np.ndarray) -> tuple[float, np.ndarray]:
        theta = torch.tensor(theta_array, dtype=torch.float64, requires_grad=True)
        loss = train_loss(theta)
        (gradient,) = torch.autograd.grad(loss, theta)
        return float(loss.detach()), gradient.numpy().astype(np.float64)

    restarts: list[FitRestart] = []
    for index, initial in enumerate(initials):
        outcome = minimize(
            objective,
            np.asarray(initial, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-12},
        )
        final = tuple(float(v) for v in outcome.x)
        with torch.no_grad():
            val = float(validation_loss(torch.tensor(final, dtype=torch.float64)))
            train = float(train_loss(torch.tensor(final, dtype=torch.float64)))
        restarts.append(
            FitRestart(
                restart_index=index,
                initial_theta=initial,
                final_theta=final,
                train_loss=train,
                validation_loss=val,
                converged=bool(outcome.success),
                iterations=int(outcome.nit),
            )
        )

    # Lowest validation loss; ties resolve to the smallest restart index.
    selected = min(restarts, key=lambda r: (r.validation_loss, r.restart_index))
    theta = selected.final_theta
    assert len(theta) == 5
    hits = tuple(
        name
        for name, value, (low, high) in zip(PARAMETER_ORDER, theta, bounds, strict=True)
        if value <= low + 1e-12 or value >= high - 1e-12
    )
    return FitResult(
        theta=(theta[0], theta[1], theta[2], theta[3], theta[4]),
        selected_restart=selected.restart_index,
        restarts=tuple(restarts),
        boundary_hits=hits,
    )
