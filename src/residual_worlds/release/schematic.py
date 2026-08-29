"""Generator for the checked-in schematic fixture bundle.

The website and report need realistic-shaped data long before any
experiment runs. This fixture is schema-identical to a real bundle but
carries ``content_status: schematic``, obviously placeholder values,
and a repeated ``SCHEMATIC DATA - NOT RESULTS`` watermark on every
figure. The production verifier rejects it wherever ``final`` is
required, and golden tests keep it impossible to confuse with evidence.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from residual_worlds.provenance import write_artifact

WATERMARK = "SCHEMATIC DATA - NOT RESULTS"

_METHODS = ("nominal", "fitted_physics", "blackbox", "residual", "oracle")


def _watermark(figure: Any) -> None:
    figure.text(
        0.5, 0.5, WATERMARK, fontsize=22, color="#C94F4F", alpha=0.35,
        ha="center", va="center", rotation=20,
    )


def _schematic_figures(directory: Path) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    rng = np.random.Generator(np.random.PCG64DXSM(7))

    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    diffs = rng.normal(0.12, 0.08, size=8)
    axis.scatter(np.linspace(-0.08, 0.08, 8), diffs, color="#7E57C2", marker="D")
    axis.errorbar([0.35], [0.12], yerr=[[0.10], [0.10]], fmt="o", color="#202124",
                  capsize=5)
    axis.axhline(0.0, color="#9A9A94", linewidth=1)
    axis.set_ylabel("paired success difference (placeholder)")
    axis.set_xticks([])
    axis.set_title("Primary contrast (schematic)")
    _watermark(figure)
    figure.savefig(directory / "F03_primary_effect.svg")
    figure.savefig(directory / "F03_primary_effect@2x.png", dpi=200)
    plt.close(figure)
    (directory / "F03_primary_effect.data.json").write_text(
        json.dumps({"watermark": WATERMARK, "pipeline_differences": diffs.tolist()})
        + "\n",
        encoding="utf-8",
    )
    registry.append(
        {
            "figure_id": "F03_primary_effect",
            "badge": "Schematic explanation",
            "caption": "Placeholder layout of the primary paired-effect figure.",
            "source": "schematic",
            "files": [
                "F03_primary_effect.svg",
                "F03_primary_effect@2x.png",
                "F03_primary_effect.data.json",
            ],
        }
    )

    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    budgets = [256, 1024, 2048, 8192, 16384]
    for method, base, color in (
        ("fitted_physics", 0.45, "#009E73"),
        ("blackbox", 0.35, "#0072B2"),
        ("residual", 0.55, "#7E57C2"),
    ):
        curve = np.clip(base + 0.12 * np.log10(np.asarray(budgets) / 256.0), 0, 1)
        axis.plot(budgets, curve, marker="o", color=color, label=method)
    axis.set_xscale("log")
    axis.set_ylim(0, 1)
    axis.set_xlabel("adaptation transitions (placeholder)")
    axis.set_ylabel("task success rate")
    axis.legend(fontsize=7)
    axis.set_title("Budget curve (schematic)")
    _watermark(figure)
    figure.savefig(directory / "F04_budget_curve.svg")
    figure.savefig(directory / "F04_budget_curve@2x.png", dpi=200)
    plt.close(figure)
    (directory / "F04_budget_curve.data.json").write_text(
        json.dumps({"watermark": WATERMARK}) + "\n", encoding="utf-8"
    )
    registry.append(
        {
            "figure_id": "F04_budget_curve",
            "badge": "Schematic explanation",
            "caption": "Placeholder layout of the budget-curve figure.",
            "source": "schematic",
            "files": [
                "F04_budget_curve.svg",
                "F04_budget_curve@2x.png",
                "F04_budget_curve.data.json",
            ],
        }
    )
    return registry


def build_schematic_fixture(destination: Path) -> Path:
    """Write the fixture bundle (deletes and recreates ``destination``)."""
    if destination.exists():
        shutil.rmtree(destination)

    key_results: dict[str, Any] = {
        "schema": 1,
        "content_status": "schematic",
        "interpretation_state": "no_results",
        "protocol_tag": "schematic-fixture",
        "analysis_id": "schematic-analysis-placeholder",
        "watermark": WATERMARK,
        "primary": {
            "contrast_id": "residual_vs_blackbox_success",
            "estimate": 0.12,
            "lower_95": 0.02,
            "upper_95": 0.22,
            "sign_flip_p": 0.5,
            "holm_adjusted_p": None,
            "practical_threshold": 0.10,
            "pipelines": 8,
            "scenarios": 24,
            "world_id": "composite_standard",
            "budget": 2048,
        },
        "secondary": [],
    }
    rng = np.random.Generator(np.random.PCG64DXSM(11))
    method_rows = [
        {
            "method": method,
            "world_id": "composite_standard",
            "budget": 2048,
            "episodes": 192,
            "successes": int(round(rate * 192)),
            "success_rate": rate,
            "collision": 8,
            "hard_limit": 6,
            "timeout": 190 - int(round(rate * 192)) - 14,
            "nonfinite": 1,
            "training_failed": 1,
            "mean_completion_time_restricted_s": 6.0,
        }
        for method, rate in zip(_METHODS, (0.30, 0.45, 0.40, 0.55, 0.80), strict=True)
    ]

    def populate(directory: Path) -> None:
        for sub in ("summary", "data", "figures", "tables"):
            (directory / sub).mkdir()

        def dump(relative: str, payload: Any) -> None:
            (directory / relative).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        dump("summary/key_results.json", key_results)
        dump(
            "summary/study_summary.json",
            {
                "title": "Residual Worlds",
                "tagline": "Keep the physics. Learn the mismatch.",
                "question": "Placeholder question text.",
                "watermark": WATERMARK,
                "design": {
                    "methods": list(_METHODS),
                    "primary_world": "composite_standard",
                    "primary_budget": 2048,
                    "pipelines": 8,
                    "scenarios": 24,
                },
                "status_note": WATERMARK,
            },
        )
        dump("summary/limitations.json", {"items": [WATERMARK]})
        dump(
            "summary/primary_contrast.json",
            {**key_results["primary"], "interpretation_state": "no_results"},
        )
        dump(
            "data/availability.json",
            {
                "evaluated_control_cells": [
                    {"world_id": "composite_standard", "budget": b}
                    for b in (256, 2048, 16384)
                ],
                "prediction_available": True,
                "note": WATERMARK,
            },
        )
        dump(
            "data/primary_effects.json",
            {
                "contrasts": [key_results["primary"]],
                "primary_cell_methods": method_rows,
                "pipeline_differences": rng.normal(0.12, 0.08, size=8).tolist(),
                "watermark": WATERMARK,
            },
        )
        dump(
            "data/budget_curve.json",
            {
                "world_id": "composite_standard",
                "success_axis": [0.0, 1.0],
                "series": {
                    method: [
                        {"budget": b,
                         "success_rate": min(1.0, 0.3 + 0.1 * i + 0.05 * j),
                         "episodes": 48}
                        for j, b in enumerate((256, 2048, 16384))
                    ]
                    for i, method in enumerate(("fitted_physics", "blackbox", "residual"))
                },
                "note": WATERMARK,
            },
        )
        dump("data/success_matrix.json", {"rows": method_rows, "watermark": WATERMARK})
        dump(
            "data/failures.json",
            {
                method: {
                    "successes": row["successes"], "collision": row["collision"],
                    "hard_limit": row["hard_limit"], "timeout": row["timeout"],
                    "nonfinite": row["nonfinite"],
                    "training_failed": row["training_failed"],
                    "episodes": row["episodes"],
                }
                for method, row in zip(_METHODS, method_rows, strict=True)
            },
        )
        dump(
            "data/evidence_grid.json",
            {
                "rows": [
                    {
                        "job_id": f"schematic-{method}-{index}",
                        "method": method,
                        "world_id": "composite_standard",
                        "budget": 2048,
                        "replicate": index % 8,
                        "scenario_id": f"schematic-scenario-{index % 24}",
                        "success": bool(rng.uniform() < 0.5),
                        "termination_code": "TIMEOUT_PARTIAL_TARGETS",
                    }
                    for method in _METHODS
                    for index in range(8)
                ],
                "watermark": WATERMARK,
            },
        )
        registry = _schematic_figures(directory / "figures")
        (directory / "figures" / "figure_registry.json").write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (directory / "tables" / "interpretation.json").write_text(
            json.dumps({"state": "no_results", "watermark": WATERMARK}, indent=2) + "\n",
            encoding="utf-8",
        )

    write_artifact(
        destination,
        "core_result_bundle",
        {
            "core_id": "schematic-fixture",
            "analysis_id": "schematic-analysis-placeholder",
            "content_status": "schematic",
            "interpretation_state": "no_results",
        },
        {"analysis": "schematic-analysis-placeholder"},
        populate,
    )
    return destination
