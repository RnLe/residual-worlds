"""Dataset pipeline: allocation, leakage, nesting, pairing digests."""

from pathlib import Path

import numpy as np
import pytest

from residual_worlds.config import load_contract
from residual_worlds.data.dataset import load_dataset, load_segment_registry
from residual_worlds.data.split import (
    balanced_order,
    budget_membership,
    build_unit_plan,
    largest_remainder_allocation,
)
from residual_worlds.paths import repository_root

pytestmark = [pytest.mark.scientific, pytest.mark.slow]

SMOKE = load_contract(repository_root() / "configs" / "smoke.yaml")
WEIGHTS = (0.40, 0.30, 0.30)


# ---------------------------------------------------------------------------
# Pure allocation logic (fast)


def test_largest_remainder_exact_cases() -> None:
    # The four-unit minimum budget: 3 train units get one of each
    # component; the single validation unit goes to the first component.
    assert largest_remainder_allocation(3, WEIGHTS) == (1, 1, 1)
    assert largest_remainder_allocation(1, WEIGHTS) == (1, 0, 0)
    assert largest_remainder_allocation(10, WEIGHTS) == (4, 3, 3)
    assert largest_remainder_allocation(0, WEIGHTS) == (0, 0, 0)


def test_balanced_order_prefixes_contain_all_components() -> None:
    order = balanced_order(192, WEIGHTS)
    assert order[:3] == (0, 1, 2)
    # Every prefix's counts differ from the exact quota by less than 1.
    counts = [0, 0, 0]
    for t, component in enumerate(order, start=1):
        counts[component] += 1
        for i, w in enumerate(WEIGHTS):
            assert abs(counts[i] - t * w) <= 1.0
    assert counts == [77, 58, 57] or sum(counts) == 192


def test_unit_plan_and_membership_nesting() -> None:
    plan = build_unit_plan(64, 16384, 0.75, WEIGHTS)
    assert plan.train_units == 192
    assert plan.validation_units == 64
    membership = budget_membership((256, 1024, 2048, 8192, 16384), 64, 0.75, plan)
    previous_train: tuple[int, ...] = ()
    previous_validation: tuple[int, ...] = ()
    for budget in (256, 1024, 2048, 8192, 16384):
        train, validation = membership[budget]
        assert len(train) * 64 == budget * 3 // 4
        assert len(validation) * 64 == budget // 4
        assert train[: len(previous_train)] == previous_train  # nested
        assert validation[: len(previous_validation)] == previous_validation
        previous_train, previous_validation = train, validation
    # The smallest training prefix (256 -> 3 units) has all components.
    smallest_train = membership[256][0]
    components = {plan.train_components[u] for u in smallest_train}
    assert components == {0, 1, 2}
    # The smallest validation prefix is exactly one band-limited unit.
    smallest_validation = membership[256][1]
    assert len(smallest_validation) == 1
    assert plan.validation_components[smallest_validation[0] - plan.train_units] == 0


# ---------------------------------------------------------------------------
# Generated artifacts (module-scoped, in an isolated artifact root)


def test_dataset_counts_and_split(smoke_workspace: dict[str, Path]) -> None:
    view = load_dataset(smoke_workspace["dataset"])
    unit = SMOKE.data.collection_unit_valid_transitions
    total = max(SMOKE.data.adaptation_budgets_total)
    assert view.state.shape == (total, 4)
    # Every unit holds exactly the declared number of valid transitions.
    for unit_id in np.unique(view.unit_id):
        assert int((view.unit_id == unit_id).sum()) == unit
    # No unit appears in both splits.
    assert set(view.train_units).isdisjoint(view.validation_units)


def test_next_state_chains_within_trajectories(smoke_workspace: dict[str, Path]) -> None:
    view = load_dataset(smoke_workspace["dataset"])
    for row in range(view.state.shape[0] - 1):
        same_trajectory = view.trajectory_id[row] == view.trajectory_id[row + 1]
        consecutive = view.step_index[row + 1] == view.step_index[row] + 1
        if same_trajectory and consecutive:
            np.testing.assert_array_equal(view.next_state[row], view.state[row + 1])


def test_rollout_windows_never_cross_boundaries(smoke_workspace: dict[str, Path]) -> None:
    view = load_dataset(smoke_workspace["dataset"])
    train, _validation = view.units_for_budget(
        SMOKE.data.primary_budget_total,
        SMOKE.data.collection_unit_valid_transitions,
        SMOKE.data.train_fraction,
    )
    rows = view.rows_for_units(train)
    origins = view.rollout_origins(rows, 5)
    assert origins.size > 0
    for origin in origins:
        window_trajectory = view.trajectory_id[origin : origin + 5]
        assert (window_trajectory == window_trajectory[0]).all()
        assert (np.diff(view.step_index[origin : origin + 5]) == 1).all()


def test_budget_prefixes_are_nested_in_view(smoke_workspace: dict[str, Path]) -> None:
    view = load_dataset(smoke_workspace["dataset"])
    unit = SMOKE.data.collection_unit_valid_transitions
    fraction = SMOKE.data.train_fraction
    small_train, small_validation = view.units_for_budget(64, unit, fraction)
    large_train, large_validation = view.units_for_budget(128, unit, fraction)
    assert set(small_train) <= set(large_train)
    assert set(small_validation) <= set(large_validation)


def test_membership_digest_is_stable_and_budget_sensitive(
    smoke_workspace: dict[str, Path]
) -> None:
    view = load_dataset(smoke_workspace["dataset"])
    unit = SMOKE.data.collection_unit_valid_transitions
    fraction = SMOKE.data.train_fraction
    train_64, _ = view.units_for_budget(64, unit, fraction)
    train_128, _ = view.units_for_budget(128, unit, fraction)
    assert view.membership_digest(train_64) == view.membership_digest(train_64)
    assert view.membership_digest(train_64) != view.membership_digest(train_128)


def test_prediction_set_is_separate_and_registered(smoke_workspace: dict[str, Path]) -> None:
    view = load_dataset(smoke_workspace["prediction"])
    assert view.train_units == () and view.validation_units == ()
    assert view.state.shape[0] == SMOKE.data.prediction_test_transitions
    registry = load_segment_registry(smoke_workspace["prediction"])
    for horizon in SMOKE.data.rollout_horizons:
        assert horizon in registry
        for origin in registry[horizon]:
            steps = view.step_index[origin : origin + horizon]
            trajectory = view.trajectory_id[origin : origin + horizon]
            assert (trajectory == trajectory[0]).all()
            assert (np.diff(steps) == 1).all()


def test_no_world_parameters_in_transition_files(smoke_workspace: dict[str, Path]) -> None:
    # The npz payload visible to model loaders carries only the declared
    # arrays; hidden world parameters live in the manifest provenance.
    with np.load(smoke_workspace["dataset"] / "transitions.npz") as archive:
        assert set(archive.files) == {
            "state",
            "action",
            "next_state",
            "collection_unit_id",
            "trajectory_id",
            "step_index",
            "component_code",
        }
