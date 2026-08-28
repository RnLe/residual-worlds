"""Cross-entropy method over latent action-knot sequences.

CEM maintains a Gaussian in unconstrained latent space; candidates are
transformed through ``u = u_max * tanh(z)`` so torque bounds hold by
construction. Deterministic numerical conventions are part of the
scientific contract, because every method must face bit-identical
optimizer behavior given the same primitive noise:

* all planner tensors are float32;
* candidate ranking is a stable ascending sort of the tuple
  (invalid flag, total cost, candidate index); non-finite costs order
  after every finite cost, then by index;
* elite standard deviation uses the population divisor (correction=0)
  with the frozen floor;
* the retention update is mu <- r mu + (1 - r) mu_elite (same for
  sigma, then floored);
* after the last iteration the final latent mean is always evaluated
  as one additional deterministic candidate.

The optimizer knows nothing about dynamics or costs; it receives a
scoring closure, so the identical code serves toy problems, tests, and
the full ensemble controller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from residual_worlds.config import PlanningConfig

# A scorer maps latent candidates [K, knots, 2] -> (cost [K], invalid [K]).
Scorer = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class CEMSettings:
    candidates: int
    elites: int
    iterations: int
    action_knots: int
    initial_latent_std: float
    latent_std_floor: float
    old_distribution_retention: float

    @staticmethod
    def from_planning(planning: PlanningConfig) -> CEMSettings:
        return CEMSettings(
            candidates=planning.candidates,
            elites=planning.elites,
            iterations=planning.iterations,
            action_knots=planning.action_knots,
            initial_latent_std=planning.initial_latent_std,
            latent_std_floor=planning.latent_std_floor,
            old_distribution_retention=planning.old_distribution_retention,
        )


@dataclass(frozen=True)
class CEMResult:
    final_mean: torch.Tensor  # [knots, 2] latent
    final_mean_cost: float
    final_mean_invalid: bool
    elite_indices: torch.Tensor  # [iterations, elites] int64
    best_cost_by_iteration: torch.Tensor  # [iterations] float32
    invalid_fraction_by_iteration: torch.Tensor  # [iterations] float32
    mean_by_iteration: torch.Tensor  # [iterations, knots, 2]
    std_by_iteration: torch.Tensor  # [iterations, knots, 2]


def rank_candidates(costs: torch.Tensor, invalid: torch.Tensor) -> torch.Tensor:
    """Stable ascending order by (invalid, cost, index); NaN after finite."""
    count = costs.shape[0]
    finite = torch.isfinite(costs)
    # Replace non-finite costs by +inf so they sort last within their
    # invalid class; stable sort preserves index order among ties.
    safe = torch.where(finite, costs, torch.full_like(costs, float("inf")))
    # Two-key stable sort: sort by cost first, then stably by invalid flag.
    by_cost = torch.argsort(safe, stable=True)
    invalid_ordered = invalid.to(torch.uint8)[by_cost]
    by_invalid = torch.argsort(invalid_ordered, stable=True)
    order: torch.Tensor = by_cost[by_invalid]
    assert order.shape[0] == count
    return order


def cem_optimize(
    scorer: Scorer,
    initial_mean: torch.Tensor,
    base_noise: torch.Tensor,
    settings: CEMSettings,
) -> CEMResult:
    """Run the frozen CEM iteration count from ``initial_mean`` (latent).

    ``base_noise`` has shape [iterations, candidates, knots, 2]; it is
    the method-independent primitive randomness, shared across paired
    methods for the same planning call.
    """
    expected = (
        settings.iterations,
        settings.candidates,
        settings.action_knots,
        2,
    )
    if tuple(base_noise.shape) != expected:
        raise ValueError(f"base noise shape {tuple(base_noise.shape)} != {expected}")

    mean = initial_mean.to(torch.float32).clone()
    std = torch.full_like(mean, settings.initial_latent_std)
    retention = settings.old_distribution_retention

    elite_indices = torch.zeros(
        (settings.iterations, settings.elites), dtype=torch.int64
    )
    best_costs = torch.zeros(settings.iterations, dtype=torch.float32)
    invalid_fractions = torch.zeros(settings.iterations, dtype=torch.float32)
    means = torch.zeros((settings.iterations,) + tuple(mean.shape), dtype=torch.float32)
    stds = torch.zeros_like(means)

    for iteration in range(settings.iterations):
        noise = base_noise[iteration].to(torch.float32)
        candidates = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        costs, invalid = scorer(candidates)
        order = rank_candidates(costs, invalid)
        elites = order[: settings.elites]
        elite_candidates = candidates[elites]
        elite_mean = elite_candidates.mean(dim=0)
        elite_std = elite_candidates.std(dim=0, correction=0)
        mean = retention * mean + (1.0 - retention) * elite_mean
        std = torch.clamp(
            retention * std + (1.0 - retention) * elite_std,
            min=settings.latent_std_floor,
        )
        elite_indices[iteration] = elites
        best_costs[iteration] = costs[order[0]]
        invalid_fractions[iteration] = invalid.to(torch.float32).mean()
        means[iteration] = mean
        stds[iteration] = std

    final_cost, final_invalid = scorer(mean.unsqueeze(0))
    return CEMResult(
        final_mean=mean,
        final_mean_cost=float(final_cost[0]),
        final_mean_invalid=bool(final_invalid[0]),
        elite_indices=elite_indices,
        best_cost_by_iteration=best_costs,
        invalid_fraction_by_iteration=invalid_fractions,
        mean_by_iteration=means,
        std_by_iteration=stds,
    )


def latent_to_action(latent: torch.Tensor, torque_limit: torch.Tensor) -> torch.Tensor:
    """Smooth bounded transform ``u = u_max * tanh(z)``."""
    return torque_limit * torch.tanh(latent)


def expand_knots(knot_actions: torch.Tensor, horizon: int) -> torch.Tensor:
    """Piecewise-constant expansion: each knot repeats for two steps.

    ``[..., knots, 2] -> [..., horizon, 2]`` with horizon = 2 * knots.
    """
    knots = knot_actions.shape[-2]
    repeat = horizon // knots
    if knots * repeat != horizon:
        raise ValueError("horizon must be a whole multiple of the knot count")
    expanded = knot_actions.unsqueeze(-2).expand(
        *knot_actions.shape[:-2], knots, repeat, 2
    )
    return expanded.reshape(*knot_actions.shape[:-2], horizon, 2)
