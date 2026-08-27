"""Fixed-step RK4 integration of arm dynamics over one control interval.

Every model condition -- nominal, fitted, black-box, residual, and the
exact-dynamics reference -- is a continuous acceleration function
wrapped by this one integrator, so a method comparison can never turn
into an integrator comparison. The commanded action is held constant
within a control interval.

Two things matter beyond textbook RK4:

* the intermediate RK4 stage evaluations are derivative probes, not an
  ordered physical path; collision checking therefore consumes the
  *accepted substep endpoints* returned by ``rk4_substep_endpoints``,
  never the stages;
* the same code integrates float64 truth and float32 planning models;
  the dtype follows the input state.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# An acceleration function maps (state [..., 4], action [..., 2]) -> qdd [..., 2].
AccelerationFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _state_derivative(
    acceleration: AccelerationFn, state: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    qd = state[..., 2:]
    qdd = acceleration(state, action)
    return torch.cat((qd, qdd), dim=-1)


def rk4_substep_endpoints(
    acceleration: AccelerationFn,
    state: torch.Tensor,
    action: torch.Tensor,
    dt: float,
    substeps: int,
) -> torch.Tensor:
    """Integrate one control interval; return endpoints ``[..., substeps + 1, 4]``.

    ``endpoints[..., 0, :]`` is the initial state and
    ``endpoints[..., -1, :]`` the next control-step state. Consecutive
    endpoint pairs (positions and velocities) are the only sanctioned
    inputs to swept-collision interpolation.
    """
    if substeps < 1:
        raise ValueError("substeps must be at least 1")
    h = dt / substeps
    endpoints = [state]
    current = state
    for _ in range(substeps):
        k1 = _state_derivative(acceleration, current, action)
        k2 = _state_derivative(acceleration, current + 0.5 * h * k1, action)
        k3 = _state_derivative(acceleration, current + 0.5 * h * k2, action)
        k4 = _state_derivative(acceleration, current + h * k3, action)
        current = current + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        endpoints.append(current)
    return torch.stack(endpoints, dim=-2)


def rk4_transition(
    acceleration: AccelerationFn,
    state: torch.Tensor,
    action: torch.Tensor,
    dt: float,
    substeps: int,
) -> torch.Tensor:
    """Next control-step state ``[..., 4]`` (last accepted substep endpoint)."""
    endpoints = rk4_substep_endpoints(acceleration, state, action, dt, substeps)
    return endpoints[..., -1, :]
