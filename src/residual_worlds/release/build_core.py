"""Build the public result bundle from one immutable analysis artifact.

The builder redacts and restructures -- it never recomputes. Estimates,
intervals, counts, and interpretation come from the analysis tables;
figures are copied with their registry; the browser-facing JSON files
carry exactly the values the site is allowed to display, each traceable
to its analysis source.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from residual_worlds.config import ExperimentContract
from residual_worlds.identity import content_id
from residual_worlds.paths import core_result_dir
from residual_worlds.provenance import is_complete, verify_artifact, write_artifact
from residual_worlds.release.schema import CONTENT_STATUSES


def build_core_bundle(
    contract: ExperimentContract,
    analysis_directory: Path,
    figures_directory: Path,
    content_status: str,
) -> dict[str, Any]:
    if content_status not in CONTENT_STATUSES:
        raise ValueError(f"unknown content status {content_status!r}")
    verify_artifact(analysis_directory)
    interpretation = json.loads(
        (analysis_directory / "interpretation.json").read_text(encoding="utf-8")
    )
    contrasts = pq.read_table(analysis_directory / "contrasts.parquet").to_pydict()
    methods_table = pq.read_table(
        analysis_directory / "method_summaries.parquet"
    ).to_pydict()
    cells = pq.read_table(analysis_directory / "cell_outcomes.parquet").to_pydict()
    prediction_path = analysis_directory / "prediction_summaries.parquet"
    predictions = (
        pq.read_table(prediction_path).to_pydict() if prediction_path.exists() else None
    )

    analysis_id = analysis_directory.name
    core_spec = {
        "schema": 1,
        "analysis_id": analysis_id,
        "content_status": content_status,
        "protocol_tag": contract.protocol.version,
    }
    core_id = content_id("core_result", core_spec)
    destination = core_result_dir(core_id)
    if is_complete(destination):
        verify_artifact(destination)
        return {"core_id": core_id, "artifact": str(destination), "reused": True}

    evaluation = contract.evaluation

    def _contrast_rows(role: str) -> list[dict[str, Any]]:
        rows = []
        for i in range(len(contrasts["contrast_id"])):
            if contrasts["role"][i] != role:
                continue
            rows.append(
                {
                    "contrast_id": contrasts["contrast_id"][i],
                    "estimate": contrasts["estimate"][i],
                    "lower_95": contrasts["lower_95"][i],
                    "upper_95": contrasts["upper_95"][i],
                    "sign_flip_p": contrasts["sign_flip_p"][i],
                    "holm_adjusted_p": contrasts["holm_adjusted_p"][i],
                    "pipelines": contrasts["pipelines"][i],
                    "scenarios": contrasts["scenarios"][i],
                }
            )
        return rows

    primary_rows = _contrast_rows("primary")
    assert len(primary_rows) == 1
    key_results = {
        "schema": 1,
        "content_status": content_status,
        "interpretation_state": interpretation["state"],
        "protocol_tag": contract.protocol.version,
        "analysis_id": analysis_id,
        "primary": {
            **primary_rows[0],
            "practical_threshold": interpretation["practical_threshold"],
            "world_id": evaluation.primary_world,
            "budget": evaluation.primary_budget,
        },
        "secondary": _contrast_rows("secondary"),
    }

    # Per-method success at the primary cell plus the full grid.
    method_rows = []
    for i in range(len(methods_table["method"])):
        method_rows.append(
            {key: methods_table[key][i] for key in methods_table}
        )
    primary_methods = [
        row
        for row in method_rows
        if row["world_id"] == evaluation.primary_world
        and row["budget"] == evaluation.primary_budget
    ]

    budget_curve: dict[str, Any] = {
        "world_id": evaluation.primary_world,
        "success_axis": [0.0, 1.0],
        "series": {},
        "note": "Only executed budgets appear; nothing is interpolated.",
    }
    for method in ("fitted_physics", "blackbox", "residual", "nominal", "oracle"):
        points = sorted(
            (
                {"budget": row["budget"], "success_rate": row["success_rate"],
                 "episodes": row["episodes"]}
                for row in method_rows
                if row["method"] == method
                and row["world_id"] == evaluation.primary_world
            ),
            key=lambda p: p["budget"],
        )
        if points:
            budget_curve["series"][method] = points

    evaluated = sorted(
        {(row["world_id"], row["budget"]) for row in method_rows}
    )
    availability = {
        "evaluated_control_cells": [
            {"world_id": world, "budget": budget} for world, budget in evaluated
        ],
        "prediction_available": predictions is not None,
        "note": "Combinations absent here were not evaluated under the frozen scope.",
    }

    failures: dict[str, dict[str, int]] = {}
    for row in method_rows:
        entry = failures.setdefault(
            row["method"],
            {"successes": 0, "collision": 0, "hard_limit": 0, "timeout": 0,
             "nonfinite": 0, "training_failed": 0, "episodes": 0},
        )
        for key in entry:
            entry[key] += row.get(key, 0)

    evidence_rows = [
        {
            "job_id": cells["job_id"][i],
            "method": cells["method"][i],
            "world_id": cells["world_id"][i],
            "budget": cells["budget"][i],
            "replicate": cells["replicate"][i],
            "scenario_id": cells["scenario_id"][i],
            "success": cells["success"][i],
            "termination_code": cells["termination_code"][i],
        }
        for i in range(len(cells["job_id"]))
    ]

    prediction_export = None
    if predictions is not None:
        prediction_export = [
            {key: predictions[key][i] for key in predictions}
            for i in range(len(predictions["method"]))
        ]

    study_summary = {
        "title": "Residual Worlds",
        "tagline": "Keep the physics. Learn the mismatch.",
        "question": (
            "Under scarce transitions from a changed simulated world, does keeping "
            "nominal rigid-body mechanics and learning only an acceleration "
            "correction support better MPC decisions than parameter fitting or "
            "learning the whole dynamics from scratch?"
        ),
        "design": {
            "methods": list(evaluation.all_reference_methods),
            "primary_world": evaluation.primary_world,
            "primary_budget": evaluation.primary_budget,
            "pipelines": evaluation.pipeline_replicates_primary,
            "scenarios": evaluation.scenario_families_primary,
        },
        "status_note": (
            "State-based simulation study. All learned methods receive identical "
            "transition data; all conditions share one frozen CEM-MPC controller."
        ),
    }
    limitations = {
        "items": [
            "Fully observed, synthetic, low-dimensional dynamics study.",
            "Target worlds are designed mismatch, not naturally occurring shift.",
            "The learned residual is an acceleration correction over the visited "
            "distribution, not a uniquely identified physical force.",
            "Ensemble spread is member disagreement, not calibrated uncertainty.",
            "The exact-dynamics reference shares the approximate planner and is "
            "not a performance ceiling.",
            "No physical-robot, perception, or sim-to-real claim.",
        ]
    }

    def populate(directory: Path) -> None:
        for sub in ("summary", "data", "figures", "tables"):
            (directory / sub).mkdir()

        def dump(relative: str, payload: Any) -> None:
            (directory / relative).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        dump("summary/key_results.json", key_results)
        dump("summary/study_summary.json", study_summary)
        dump("summary/limitations.json", limitations)
        dump(
            "summary/primary_contrast.json",
            {**key_results["primary"], "interpretation_state": interpretation["state"]},
        )
        dump("data/availability.json", availability)
        dump(
            "data/primary_effects.json",
            {
                "contrasts": primary_rows + key_results["secondary"],
                "primary_cell_methods": primary_methods,
                "pipeline_differences": _pipeline_differences(analysis_directory),
            },
        )
        dump("data/budget_curve.json", budget_curve)
        dump("data/success_matrix.json", {"rows": method_rows})
        dump("data/failures.json", failures)
        dump("data/evidence_grid.json", {"rows": evidence_rows})
        if prediction_export is not None:
            dump("data/prediction_summaries.json", {"rows": prediction_export})
        for path in sorted(figures_directory.glob("*")):
            if path.is_file():
                shutil.copy2(path, directory / "figures" / path.name)
        tables_dir = figures_directory / "tables"
        if tables_dir.exists():
            for path in sorted(tables_dir.glob("*")):
                shutil.copy2(path, directory / "tables" / path.name)

    write_artifact(
        destination,
        "core_result_bundle",
        {"core_id": core_id, "analysis_id": analysis_id,
         "content_status": content_status,
         "interpretation_state": interpretation["state"]},
        {"analysis": analysis_id},
        populate,
    )
    return {"core_id": core_id, "artifact": str(destination), "reused": False}


def _pipeline_differences(analysis_directory: Path) -> list[float]:
    with np.load(analysis_directory / "panel.npz") as panel:
        values = panel["values"]
        methods = [str(m) for m in panel["methods"]]
    residual = values[:, :, methods.index("residual")]
    blackbox = values[:, :, methods.index("blackbox")]
    return [float(v) for v in np.mean(residual - blackbox, axis=1)]
