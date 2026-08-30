"""End-to-end CPU smoke pipeline.

Runs the complete chain on the smoke configuration -- scenario banks,
datasets, prediction sets, fitted and neural training, condition
assembly, every manifest control and prediction row, completeness
verification, locked-style analysis, figures, and tables -- in minutes
on a CPU. The numbers mean nothing scientifically; the run proves that
every stage connects, every artifact validates, and the analysis
produces a templated interpretation.
"""

from __future__ import annotations

import time
from typing import Any

from residual_worlds.analysis.aggregate import run_analysis
from residual_worlds.analysis.figures import render_figures
from residual_worlds.analysis.tables import export_tables
from residual_worlds.config import ExperimentContract
from residual_worlds.data.generate import generate_prediction_set, generate_world_dataset
from residual_worlds.evaluation.manifest import build_execution_manifest
from residual_worlds.evaluation.metrics import run_prediction_job
from residual_worlds.evaluation.runner import run_control_job
from residual_worlds.evaluation.verify import verify_complete
from residual_worlds.models.condition import build_condition
from residual_worlds.paths import artifact_root, condition_dir, figures_dir
from residual_worlds.task.scenarios import generate_bank, load_bank, write_bank_manifest
from residual_worlds.training.train import fit_physics_run, train_neural_member


def run_smoke(contract: ExperimentContract) -> dict[str, Any]:
    started = time.perf_counter()
    stage_times: dict[str, float] = {}

    def mark(stage: str) -> None:
        stage_times[stage] = round(time.perf_counter() - started, 1)

    # 1. Scenario banks.
    scenario_dir = artifact_root() / "smoke_scenarios"
    for bank in ("training_task", "pilot", "protected"):
        path = scenario_dir / f"{bank}.json"
        if not path.exists():
            write_bank_manifest(contract, bank, generate_bank(contract, bank), scenario_dir)
    protected = load_bank(scenario_dir / "protected.json")
    mark("scenarios")

    # 2. Manifest expansion fixes everything that follows.
    manifest = build_execution_manifest(contract, protected)

    # 3. Datasets and prediction sets.
    datasets: dict[tuple[str, int], Any] = {}
    for world_id, replicate in manifest.dataset_jobs:
        result = generate_world_dataset(contract, world_id, replicate, scenario_dir)
        datasets[(world_id, replicate)] = result["artifact"]
    prediction_sets: dict[tuple[str, int], Any] = {}
    for world_id, replicate in manifest.prediction_set_jobs:
        result = generate_prediction_set(contract, world_id, replicate, scenario_dir)
        prediction_sets[(world_id, replicate)] = result["artifact"]
    mark("data")

    # 4. Training and fitting.
    from pathlib import Path

    training_statuses: dict[str, int] = {}
    for train_job in manifest.train_jobs:
        dataset_path = Path(datasets[(train_job.world_id, train_job.replicate)])
        if train_job.method == "fitted_physics":
            result = fit_physics_run(
                contract, dataset_path, train_job.world_id, train_job.budget,
                train_job.replicate,
            )
        else:
            result = train_neural_member(
                contract, dataset_path, train_job.method, train_job.world_id,
                train_job.budget, train_job.replicate, train_job.member,
            )
        status = result.get("status", "READY")
        training_statuses[status] = training_statuses.get(status, 0) + 1
    mark("training")

    # 5. Conditions for every control cell.
    condition_paths: dict[tuple[str, str, int, int], Path | None] = {}
    for control_job in manifest.control_jobs:
        key = (
            control_job.method, control_job.world_id, control_job.budget,
            control_job.replicate,
        )
        if key in condition_paths:
            continue
        if control_job.method in ("nominal", "oracle"):
            condition_paths[key] = None
            continue
        cell_dataset = Path(datasets[(control_job.world_id, control_job.replicate)])
        built = build_condition(
            contract, control_job.method, control_job.world_id, control_job.budget,
            control_job.replicate, cell_dataset.name,
        )
        condition_paths[key] = condition_dir(built["condition_id"])
    mark("conditions")

    # 6. Control rows in manifest order.
    scenarios_by_id = {s.scenario_id: s for s in protected}
    outcomes: dict[str, int] = {}
    for control_job in manifest.control_jobs:
        key = (
            control_job.method, control_job.world_id, control_job.budget,
            control_job.replicate,
        )
        cell_dataset_path: Path | None = (
            Path(datasets[(control_job.world_id, control_job.replicate)])
            if control_job.method in ("fitted_physics", "blackbox", "residual")
            else None
        )
        row = run_control_job(
            contract, control_job, scenarios_by_id[control_job.scenario_id],
            condition_paths[key], cell_dataset_path,
        )
        outcomes[row["termination"]] = outcomes.get(row["termination"], 0) + 1
    mark("control")

    # 7. Prediction rows.
    for prediction_job in manifest.prediction_jobs:
        prediction_dataset = Path(datasets[(prediction_job.world_id, prediction_job.replicate)])
        built = build_condition(
            contract, prediction_job.method, prediction_job.world_id,
            prediction_job.budget, prediction_job.replicate, prediction_dataset.name,
        )
        run_prediction_job(
            contract, prediction_job, condition_dir(built["condition_id"]),
            Path(prediction_sets[(prediction_job.world_id, prediction_job.replicate)]),
        )
    mark("prediction")

    # 8. Completeness, analysis, figures, tables.
    report = verify_complete(manifest)
    if not report.complete:
        return {
            "ok": False,
            "stage_times_s": stage_times,
            "missing": list(report.missing)[:10],
            "corrupt": list(report.corrupt)[:10],
        }
    analysis = run_analysis(contract, manifest, protected)
    analysis_directory = Path(analysis["artifact"])
    figures = render_figures(
        contract, analysis_directory, figures_dir(analysis["analysis_id"])
    )
    tables = export_tables(
        analysis_directory, figures_dir(analysis["analysis_id"]) / "tables"
    )
    mark("analysis")

    return {
        "ok": True,
        "stage_times_s": stage_times,
        "control_rows": report.control_complete,
        "prediction_rows": report.prediction_complete,
        "training_statuses": training_statuses,
        "termination_codes": dict(sorted(outcomes.items())),
        "interpretation": analysis["interpretation"],
        "figures": figures["figures"],
        "tables": tables["exported"],
        "analysis_artifact": analysis["artifact"],
    }
