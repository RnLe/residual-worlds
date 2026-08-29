"""Locked analysis: from evaluation rows to estimates and interpretation.

The analysis consumes one explicit input set -- the manifest's control
and prediction jobs, each resolved to its checksum-valid artifact --
and writes one immutable analysis artifact. It refuses unexpected
missing or duplicate cells; a TRAINING_FAILED row participates as
success 0 with its timeout-restricted completion time, exactly as
scheduled. Nothing downstream (figures, tables, report, site) computes
statistics again; they read this artifact.

The confirmatory family is deliberately small: the primary
residual-minus-black-box success contrast with its crossed bootstrap
interval and practical threshold, plus the two declared secondary
contrasts with Holm-adjusted sign-flip p-values as the labeled
assumption-dependent sensitivity check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from residual_worlds.analysis.statistics import (
    PairedPanel,
    crossed_stratified_bootstrap,
    exact_sign_flip_p_value,
    holm_adjust,
    interpretation_state,
)
from residual_worlds.config import ExperimentContract
from residual_worlds.evaluation.manifest import ExecutionManifest
from residual_worlds.identity import content_id
from residual_worlds.paths import analysis_dir, evaluation_dir, prediction_job_dir
from residual_worlds.provenance import (
    is_complete,
    verify_artifact,
    write_artifact,
)
from residual_worlds.types import Scenario


class AnalysisInputError(RuntimeError):
    """A scheduled row is missing, duplicated, or corrupt."""


def _load_cell_outcomes(manifest: ExecutionManifest) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for job in manifest.control_jobs:
        if job.job_id in seen:
            raise AnalysisInputError(f"duplicate scheduled control job {job.job_id}")
        seen.add(job.job_id)
        directory = evaluation_dir(job.job_id)
        if not is_complete(directory):
            raise AnalysisInputError(f"missing control artifact for job {job.job_id}")
        verify_artifact(directory)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        outcomes.append(summary)
    return outcomes


def _load_prediction_summaries(manifest: ExecutionManifest) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for job in manifest.prediction_jobs:
        directory = prediction_job_dir(job.job_id)
        if not is_complete(directory):
            raise AnalysisInputError(f"missing prediction artifact for job {job.job_id}")
        verify_artifact(directory)
        summaries.append(
            json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        )
    return summaries


def _primary_panel(
    contract: ExperimentContract,
    outcomes: list[dict[str, Any]],
    scenarios: list[Scenario],
) -> PairedPanel:
    evaluation = contract.evaluation
    methods = tuple(evaluation.all_reference_methods)
    replicates = sorted(
        {
            row["replicate"]
            for row in outcomes
            if row["world_id"] == evaluation.primary_world
            and row["budget"] == evaluation.primary_budget
        }
    )
    scenario_ids = [
        s.scenario_id
        for s in scenarios[: evaluation.scenario_families_primary]
    ]
    strata = tuple(
        s.stratum_id for s in scenarios[: evaluation.scenario_families_primary]
    )
    values = np.full(
        (len(replicates), len(scenario_ids), len(methods)), np.nan, dtype=np.float64
    )
    for row in outcomes:
        if (
            row["world_id"] != evaluation.primary_world
            or row["budget"] != evaluation.primary_budget
        ):
            continue
        r = replicates.index(row["replicate"])
        if row["scenario_id"] not in scenario_ids:
            continue
        s = scenario_ids.index(row["scenario_id"])
        m = methods.index(row["method"])
        if not np.isnan(values[r, s, m]):
            raise AnalysisInputError(
                f"duplicate primary cell ({row['replicate']}, {row['scenario_id']}, "
                f"{row['method']})"
            )
        values[r, s, m] = 1.0 if row["success"] else 0.0
    if np.isnan(values).any():
        missing = int(np.isnan(values).sum())
        raise AnalysisInputError(f"{missing} primary cells missing from evaluation rows")
    return PairedPanel(values=values, methods=methods, scenario_strata=strata)


def run_analysis(
    contract: ExperimentContract,
    manifest: ExecutionManifest,
    scenarios: list[Scenario],
) -> dict[str, Any]:
    outcomes = _load_cell_outcomes(manifest)
    prediction_summaries = _load_prediction_summaries(manifest)

    analysis_spec = {
        "schema": 1,
        "root_seed": contract.numerics.root_seed,
        "control_jobs": sorted(job.job_id for job in manifest.control_jobs),
        "prediction_jobs": sorted(job.job_id for job in manifest.prediction_jobs),
        "primary_contrast": contract.analysis.primary_contrast.contrast_id,
        "bootstrap_replicates": contract.analysis.bootstrap_replicates,
    }
    analysis_id = content_id("analysis", analysis_spec)
    destination = analysis_dir(analysis_id)
    if is_complete(destination):
        verify_artifact(destination)
        interpretation = json.loads(
            (destination / "interpretation.json").read_text(encoding="utf-8")
        )
        return {
            "analysis_id": analysis_id,
            "artifact": str(destination),
            "reused": True,
            "interpretation": interpretation,
        }

    panel = _primary_panel(contract, outcomes, scenarios)
    analysis = contract.analysis
    replicates = analysis.bootstrap_replicates

    def _contrast_pair(weights: dict[str, float]) -> tuple[str, str]:
        positive = next(m for m, w in weights.items() if w > 0)
        negative = next(m for m, w in weights.items() if w < 0)
        return positive, negative

    contrast_rows: list[dict[str, Any]] = []
    sign_p: dict[str, float] = {}
    primary_a, primary_b = _contrast_pair(analysis.primary_contrast.weights)
    primary = crossed_stratified_bootstrap(
        panel, primary_a, primary_b, replicates, contract.numerics.root_seed,
        analysis.primary_contrast.contrast_id,
    )
    primary_sign_p = exact_sign_flip_p_value(
        panel.pipeline_differences(primary_a, primary_b)
    )
    contrast_rows.append(
        {
            "contrast_id": analysis.primary_contrast.contrast_id,
            "role": "primary",
            "estimate": primary.estimate,
            "lower_95": primary.lower,
            "upper_95": primary.upper,
            "sign_flip_p": primary_sign_p,
            "holm_adjusted_p": None,
            "pipelines": panel.values.shape[0],
            "scenarios": panel.values.shape[1],
        }
    )
    for contrast in analysis.secondary_confirmatory_contrasts:
        a, b = _contrast_pair(contrast.weights)
        result = crossed_stratified_bootstrap(
            panel, a, b, replicates, contract.numerics.root_seed, contrast.contrast_id
        )
        p = exact_sign_flip_p_value(panel.pipeline_differences(a, b))
        sign_p[contrast.contrast_id] = p
        contrast_rows.append(
            {
                "contrast_id": contrast.contrast_id,
                "role": "secondary",
                "estimate": result.estimate,
                "lower_95": result.lower,
                "upper_95": result.upper,
                "sign_flip_p": p,
                "holm_adjusted_p": None,
                "pipelines": panel.values.shape[0],
                "scenarios": panel.values.shape[1],
            }
        )
    adjusted = holm_adjust(sign_p) if sign_p else {}
    for row in contrast_rows:
        if row["role"] == "secondary":
            row["holm_adjusted_p"] = adjusted[row["contrast_id"]]

    state = interpretation_state(
        primary, analysis.practical_effect_threshold_absolute, any_rows=bool(outcomes)
    )
    interpretation = {
        "state": state,
        "primary_contrast": analysis.primary_contrast.contrast_id,
        "estimate": primary.estimate,
        "lower_95": primary.lower,
        "upper_95": primary.upper,
        "practical_threshold": analysis.practical_effect_threshold_absolute,
        "sign_flip_p_sensitivity": primary_sign_p,
        "pipelines": panel.values.shape[0],
        "scenarios": panel.values.shape[1],
    }

    # Descriptive per-method summaries across every (world, budget) cell.
    method_rows: dict[str, list[Any]] = {
        "method": [], "world_id": [], "budget": [], "episodes": [], "successes": [],
        "success_rate": [], "collision": [], "hard_limit": [], "timeout": [],
        "nonfinite": [], "training_failed": [],
        "mean_completion_time_restricted_s": [],
    }
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in outcomes:
        groups.setdefault((row["method"], row["world_id"], row["budget"]), []).append(row)
    for (method, world_id, budget), rows in sorted(groups.items()):
        method_rows["method"].append(method)
        method_rows["world_id"].append(world_id)
        method_rows["budget"].append(budget)
        method_rows["episodes"].append(len(rows))
        method_rows["successes"].append(sum(1 for r in rows if r["success"]))
        method_rows["success_rate"].append(
            sum(1 for r in rows if r["success"]) / len(rows)
        )
        codes = [r["termination_code"] for r in rows]
        method_rows["collision"].append(codes.count("OBSTACLE_COLLISION"))
        method_rows["hard_limit"].append(codes.count("HARD_LIMIT_OR_SPEED"))
        method_rows["timeout"].append(
            codes.count("TIMEOUT_ZERO_TARGETS") + codes.count("TIMEOUT_PARTIAL_TARGETS")
        )
        method_rows["nonfinite"].append(codes.count("NONFINITE_OR_MODEL_ERROR"))
        method_rows["training_failed"].append(codes.count("TRAINING_FAILED"))
        method_rows["mean_completion_time_restricted_s"].append(
            float(np.mean([r["completion_time_restricted_s"] for r in rows]))
        )

    cell_columns: dict[str, list[Any]] = {
        key: [row.get(key) for row in outcomes]
        for key in (
            "job_id", "method", "world_id", "budget", "replicate", "scenario_id",
            "success", "termination_code", "executed_steps", "targets_completed",
            "completion_time_restricted_s", "realized_cost_total",
            "predicted_cost_first_call", "torque_effort", "action_variation",
            "min_clearance_m", "support_distance_mean", "support_distance_max",
            "planning_time_ms_p50", "planning_time_ms_p95",
        )
    }
    cell_columns["panel_memberships"] = [
        ",".join(row["panel_memberships"]) for row in outcomes
    ]

    prediction_columns: dict[str, list[Any]] = {
        "method": [], "world_id": [], "budget": [], "replicate": [], "horizon": [],
        "eligible": [], "finite": [], "invalid_fraction": [], "rmse": [],
    }
    for summary in prediction_summaries:
        for horizon, entry in summary["per_horizon"].items():
            prediction_columns["method"].append(summary["method"])
            prediction_columns["world_id"].append(summary["world_id"])
            prediction_columns["budget"].append(summary["budget"])
            prediction_columns["replicate"].append(summary["replicate"])
            prediction_columns["horizon"].append(int(horizon))
            prediction_columns["eligible"].append(entry["eligible"])
            prediction_columns["finite"].append(entry["finite"])
            prediction_columns["invalid_fraction"].append(entry["invalid_fraction"])
            prediction_columns["rmse"].append(entry["rmse"])

    def populate(directory: Path) -> None:
        pq.write_table(pa.table(cell_columns), directory / "cell_outcomes.parquet")
        pq.write_table(pa.table(method_rows), directory / "method_summaries.parquet")
        pq.write_table(
            pa.table(
                {
                    key: [row[key] for row in contrast_rows]
                    for key in contrast_rows[0]
                }
            ),
            directory / "contrasts.parquet",
        )
        if prediction_columns["method"]:
            pq.write_table(
                pa.table(prediction_columns), directory / "prediction_summaries.parquet"
            )
        np.savez(
            directory / "bootstrap_draws.npz",
            primary=primary.draws,
        )
        (directory / "interpretation.json").write_text(
            json.dumps(interpretation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez(
            directory / "panel.npz",
            values=panel.values,
            scenario_strata=np.asarray(panel.scenario_strata, dtype=np.int64),
            methods=np.asarray(panel.methods, dtype="U32"),
        )

    write_artifact(
        destination,
        "analysis",
        {"analysis_id": analysis_id},
        {"control_jobs": len(manifest.control_jobs),
         "prediction_jobs": len(manifest.prediction_jobs)},
        populate,
    )
    return {
        "analysis_id": analysis_id,
        "artifact": str(destination),
        "reused": False,
        "interpretation": interpretation,
    }
