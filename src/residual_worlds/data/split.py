"""Unit allocation, balanced orderings, and nested budget membership.

The unit of splitting is one collection unit (a fixed number of valid
transitions), never a single transition: adjacent timesteps from one
trajectory are dependent, and letting them straddle train/validation
would make validation optimistic.

Two deterministic constructions live here:

* **largest-remainder allocation** of units to excitation components
  (and of budget prefixes to units), with ties broken by fixed
  component order;
* **balanced greedy orderings** whose every prefix tracks the target
  weights as closely as integer counts allow -- so nested budget
  prefixes stay component-balanced, and the smallest training prefix
  provably contains every component.
"""

from __future__ import annotations

from dataclasses import dataclass

COMPONENT_ORDER = ("band_limited_random", "multisine", "nominal_mpc")
COMPONENT_CODES = {name: code for code, name in enumerate(COMPONENT_ORDER)}


def largest_remainder_allocation(total: int, weights: tuple[float, ...]) -> tuple[int, ...]:
    """Integer allocation of ``total`` items proportional to ``weights``."""
    if total < 0:
        raise ValueError("total must be non-negative")
    quotas = [total * w for w in weights]
    floors = [int(q) for q in quotas]
    remainder = total - sum(floors)
    by_fraction = sorted(
        range(len(weights)), key=lambda i: (-(quotas[i] - floors[i]), i)
    )
    for index in by_fraction[:remainder]:
        floors[index] += 1
    return tuple(floors)


def balanced_order(total: int, weights: tuple[float, ...]) -> tuple[int, ...]:
    """Greedy low-discrepancy sequence: prefix counts track the weights.

    At each position the item with the largest deficit
    ``(t + 1) * w_i - count_i`` is chosen; ties resolve to the smallest
    index, so the sequence is fully deterministic.
    """
    counts = [0] * len(weights)
    order: list[int] = []
    for t in range(total):
        deficits = [(t + 1) * w - c for w, c in zip(weights, counts, strict=True)]
        best = max(range(len(weights)), key=lambda i: (deficits[i], -i))
        order.append(best)
        counts[best] += 1
    return tuple(order)


@dataclass(frozen=True)
class UnitPlan:
    """Deterministic layout of one dataset's collection units."""

    unit_size: int
    train_components: tuple[int, ...]  # component code per train unit, in order
    validation_components: tuple[int, ...]

    @property
    def train_units(self) -> int:
        return len(self.train_components)

    @property
    def validation_units(self) -> int:
        return len(self.validation_components)

    @property
    def total_units(self) -> int:
        return self.train_units + self.validation_units


def build_unit_plan(
    unit_size: int,
    max_budget: int,
    train_fraction: float,
    component_weights: tuple[float, float, float],
) -> UnitPlan:
    total_units = max_budget // unit_size
    train_units = round(total_units * train_fraction)
    validation_units = total_units - train_units
    return UnitPlan(
        unit_size=unit_size,
        train_components=balanced_order(train_units, component_weights),
        validation_components=balanced_order(validation_units, component_weights),
    )


def budget_unit_counts(
    budget: int, unit_size: int, train_fraction: float
) -> tuple[int, int]:
    """(train units, validation units) forming the nested prefix of ``budget``."""
    units = budget // unit_size
    train = round(units * train_fraction)
    return train, units - train


def budget_membership(
    budgets: tuple[int, ...], unit_size: int, train_fraction: float, plan: UnitPlan
) -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Per budget: (train unit indices, validation unit indices).

    Unit indices number train units ``0 .. T-1`` and validation units
    ``T .. T+V-1`` in plan order; prefixes are nested by construction.
    """
    result: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for budget in budgets:
        train_count, validation_count = budget_unit_counts(budget, unit_size, train_fraction)
        if train_count > plan.train_units or validation_count > plan.validation_units:
            raise ValueError(f"budget {budget} exceeds the generated unit plan")
        train_ids = tuple(range(train_count))
        validation_ids = tuple(
            plan.train_units + index for index in range(validation_count)
        )
        result[budget] = (train_ids, validation_ids)
    return result
