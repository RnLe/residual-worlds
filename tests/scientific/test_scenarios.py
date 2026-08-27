"""Scenario generator: determinism, stratification, structural invariants."""

import numpy as np
import pytest
import torch

from residual_worlds.config import load_contract
from residual_worlds.paths import repository_root
from residual_worlds.physics.kinematics import end_effector_position
from residual_worlds.task.geometry import arm_clearance_with_radius
from residual_worlds.task.scenarios import (
    SECTOR_ORDERS,
    generate_bank,
    generate_scenario,
    load_bank,
    stratum_assignment,
    write_bank_manifest,
)

pytestmark = pytest.mark.scientific

SMOKE = load_contract(repository_root() / "configs" / "smoke.yaml")
ROOT = SMOKE.numerics.root_seed


def test_stratum_assignment_balances_banks() -> None:
    twelve = stratum_assignment(ROOT, "pilot", 12)
    assert sorted(twelve) == list(range(12))
    twenty_four = stratum_assignment(ROOT, "protected", 24)
    assert all(twenty_four.count(s) == 2 for s in range(12))
    forty = stratum_assignment(ROOT, "calibration", 40)
    counts = [forty.count(s) for s in range(12)]
    assert max(counts) - min(counts) <= 1


def test_sector_orders_are_all_six_permutations() -> None:
    assert len(SECTOR_ORDERS) == 6
    assert len(set(SECTOR_ORDERS)) == 6


def test_generation_is_deterministic() -> None:
    a, attempts_a = generate_scenario(SMOKE, "pilot", 0, 8)
    b, attempts_b = generate_scenario(SMOKE, "pilot", 0, 8)
    assert attempts_a == attempts_b
    assert a == b


def test_different_indices_differ() -> None:
    a, _ = generate_scenario(SMOKE, "pilot", 0, 8)
    b, _ = generate_scenario(SMOKE, "pilot", 1, 8)
    assert a.scenario_id != b.scenario_id


@pytest.mark.slow
def test_generated_scenario_satisfies_structural_constraints() -> None:
    generator = SMOKE.task.scenario_generator
    arm = SMOKE.arm
    scenario, _ = generate_scenario(SMOKE, "pilot", 0, 8)

    # Stratum encoding round-trips.
    assert scenario.stratum_id == SECTOR_ORDERS.index(scenario.target_order) * 2 + (
        scenario.obstacle_chord_index
    )

    # Targets respect separation, chord length, and radial constraints.
    targets = [np.array(t) for t in scenario.targets_xy_m]
    for i in range(3):
        for j in range(i + 1, 3):
            assert (
                np.linalg.norm(targets[i] - targets[j])
                >= generator.minimum_pairwise_separation_m
            )
        assert (
            generator.radial_distance_m[0]
            <= float(np.linalg.norm(targets[i]))
            <= generator.radial_distance_m[1]
        )
    chord = sum(
        float(np.linalg.norm(b - a)) for a, b in zip(targets[:-1], targets[1:], strict=True)
    )
    low, high = generator.total_ordered_chord_length_m
    assert low <= chord <= high

    # Obstacle clears every target center by radius + margin.
    ox, oy, orad = scenario.obstacle_xy_radius_m
    for t in targets:
        assert (
            float(np.linalg.norm(t - np.array([ox, oy])))
            >= orad + generator.minimum_target_center_clearance_beyond_radius_m
        )

    # Initial pose: inside margins, clear of the obstacle, far from target 1.
    q0 = torch.tensor(scenario.initial_state[:2], dtype=torch.float64)
    clearance = arm_clearance_with_radius(
        q0,
        torch.tensor([ox, oy], dtype=torch.float64),
        orad,
        SMOKE.task.arm_safety_radius_m,
        arm,
    )
    assert float(clearance) >= generator.minimum_signed_obstacle_clearance_m
    ee0 = end_effector_position(q0, arm).numpy()
    assert (
        float(np.linalg.norm(targets[0] - ee0))
        >= generator.minimum_initial_to_first_distance_m
    )


@pytest.mark.slow
def test_bank_manifest_roundtrip(tmp_path) -> None:
    scenarios = generate_bank(SMOKE, "pilot")
    path = write_bank_manifest(SMOKE, "pilot", scenarios, tmp_path)
    loaded = load_bank(path)
    assert loaded == scenarios
