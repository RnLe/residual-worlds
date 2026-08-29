"""Integration: manifest expansion, paired control rows, prediction jobs."""

import json
from pathlib import Path

import pytest

from residual_worlds.config import load_contract
from residual_worlds.evaluation.manifest import build_execution_manifest
from residual_worlds.evaluation.metrics import run_prediction_job
from residual_worlds.evaluation.runner import run_control_job
from residual_worlds.models.condition import build_condition
from residual_worlds.paths import repository_root
from residual_worlds.task.scenarios import load_bank
from residual_worlds.training.train import fit_physics_run, train_neural_member

pytestmark = [pytest.mark.scientific, pytest.mark.slow]

SMOKE = load_contract(repository_root() / "configs" / "smoke.yaml")
WORLD = "composite_standard"
BUDGET = 128


@pytest.fixture(scope="module")
def stack(smoke_workspace: dict[str, Path]) -> dict:
    """Trained conditions plus the expanded manifest (replicate 0 only)."""
    dataset = smoke_workspace["dataset"]
    for method in ("blackbox", "residual"):
        for member in range(SMOKE.models.neural_common.ensemble_members):
            result = train_neural_member(
                SMOKE, dataset, method, WORLD, BUDGET, 0, member
            )
            assert result["status"] == "READY"
    fit_physics_run(SMOKE, dataset, WORLD, BUDGET, 0)

    conditions = {}
    for method in ("nominal", "fitted_physics", "blackbox", "residual", "oracle"):
        dataset_id = dataset.name if method not in ("nominal", "oracle") else None
        conditions[method] = build_condition(SMOKE, method, WORLD, BUDGET, 0, dataset_id)

    scenarios = load_bank(smoke_workspace["scenario_dir"] / "protected.json")
    manifest = build_execution_manifest(SMOKE, scenarios)
    return {
        "conditions": conditions,
        "manifest": manifest,
        "scenarios": {s.scenario_id: s for s in scenarios},
        "dataset": dataset,
    }


def test_manifest_counts_and_panels(stack: dict) -> None:
    manifest = stack["manifest"]
    evaluation = SMOKE.evaluation
    primary_rows = [
        j
        for j in manifest.control_jobs
        if j.budget == evaluation.primary_budget and j.world_id == WORLD
    ]
    assert len(primary_rows) == (
        evaluation.pipeline_replicates_primary
        * evaluation.scenario_families_primary
        * len(evaluation.all_reference_methods)
    )
    # Reused primary-budget subset rows carry both panel memberships.
    double = [j for j in primary_rows if len(j.panel_memberships) == 2]
    assert double
    # Endpoint rows exist only for adapted methods and carry the curve panel.
    endpoint_rows = [j for j in manifest.control_jobs if j.budget != evaluation.primary_budget]
    assert endpoint_rows
    assert all(j.method in ("fitted_physics", "blackbox", "residual") for j in endpoint_rows)
    assert all(len(j.panel_memberships) == 1 for j in endpoint_rows)


def test_manifest_order_is_deterministic(stack: dict) -> None:
    scenarios = list(stack["scenarios"].values())
    again = build_execution_manifest(SMOKE, scenarios)
    assert [j.job_id for j in again.control_jobs] == [
        j.job_id for j in stack["manifest"].control_jobs
    ]


def test_control_rows_share_primitive_noise_across_methods(stack: dict) -> None:
    manifest = stack["manifest"]
    # One primary-budget replicate-0 cell, all five methods.
    cell = {}
    scenario_id = next(
        j.scenario_id
        for j in manifest.control_jobs
        if j.replicate == 0 and j.budget == SMOKE.evaluation.primary_budget
    )
    for job in manifest.control_jobs:
        if (
            job.replicate == 0
            and job.scenario_id == scenario_id
            and job.budget == SMOKE.evaluation.primary_budget
        ):
            cell[job.method] = job
    assert set(cell) == set(SMOKE.evaluation.all_reference_methods)

    scenario = stack["scenarios"][scenario_id]
    hashes = {}
    outcomes = {}
    for method, job in cell.items():
        condition = (
            Path(stack["conditions"][method]["artifact"])
            if method not in ("nominal", "oracle")
            else None
        )
        result = run_control_job(SMOKE, job, scenario, condition, stack["dataset"])
        outcomes[method] = result
        artifact = Path(result["artifact"])
        import pyarrow.parquet as pq

        calls = pq.read_table(artifact / "planner_calls.parquet").to_pydict()
        hashes[method] = calls["noise_sha256"][0]
    # All five methods received the same first-call primitive noise.
    assert len(set(hashes.values())) == 1, hashes
    # Every row terminated with a declared code.
    for method, result in outcomes.items():
        assert result["termination"] in (
            "SUCCESS",
            "NONFINITE_OR_MODEL_ERROR",
            "HARD_LIMIT_OR_SPEED",
            "OBSTACLE_COLLISION",
            "TIMEOUT_ZERO_TARGETS",
            "TIMEOUT_PARTIAL_TARGETS",
        ), method


def test_control_row_artifacts_have_support_semantics(stack: dict) -> None:
    manifest = stack["manifest"]
    scenario_id = next(
        j.scenario_id
        for j in manifest.control_jobs
        if j.replicate == 0 and j.budget == SMOKE.evaluation.primary_budget
    )
    for job in manifest.control_jobs:
        if job.replicate != 0 or job.scenario_id != scenario_id:
            continue
        if job.budget != SMOKE.evaluation.primary_budget:
            continue
        from residual_worlds.paths import evaluation_dir

        summary = json.loads(
            (evaluation_dir(job.job_id) / "summary.json").read_text()
        )
        if job.method in ("nominal", "oracle"):
            assert summary["support_distance_mean"] is None
            assert summary["support_not_applicable_reason"] == "no_target_data_prefix"
        elif summary["executed_steps"] > 0:
            assert summary["support_distance_mean"] is not None


def test_prediction_job_produces_denominators(stack: dict) -> None:
    manifest = stack["manifest"]
    job = next(
        p
        for p in manifest.prediction_jobs
        if p.replicate == 0 and p.budget == BUDGET and p.method == "residual"
    )
    from residual_worlds.paths import prediction_set_dir  # noqa: F401

    result = run_prediction_job(
        SMOKE,
        job,
        Path(stack["conditions"]["residual"]["artifact"]),
        stack["dataset"].parent.parent / "prediction_sets" / _prediction_set_name(stack),
    )
    for horizon in SMOKE.data.rollout_horizons:
        summary = result.get("per_horizon") or json.loads(
            (Path(result["artifact"]) / "summary.json").read_text()
        )["per_horizon"]
        entry = summary[str(horizon)]
        assert entry["eligible"] > 0
        assert entry["finite"] <= entry["eligible"]
        assert entry["rmse"] is None or entry["rmse"] >= 0.0


def _prediction_set_name(stack: dict) -> str:
    root = stack["dataset"].parent.parent / "prediction_sets"
    names = [p.name for p in root.iterdir() if p.is_dir()]
    assert len(names) == 1
    return names[0]
