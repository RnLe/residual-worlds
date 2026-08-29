"""Publication figures from one immutable analysis artifact.

Figures never recompute statistics: estimates, intervals, and counts
come from the analysis tables. Each figure carries a fixed evidence
badge -- confirmatory result, diagnostic, or schematic explanation --
and every export writes SVG, a 2x PNG, the exact displayed data as
JSON, and a registry entry, so any plotted value can be traced to its
source table.

Figures whose inputs are outside the evaluated scope render an explicit
"not evaluated under the frozen scope" placeholder instead of being
silently absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from residual_worlds.config import ExperimentContract
from residual_worlds.types import METHOD_DISPLAY_NAMES

# One stable, contrast-checked palette across every artifact surface,
# with non-color cues (markers) carried alongside.
METHOD_COLORS = {
    "truth": "#202124",
    "nominal": "#A65F00",
    "fitted_physics": "#009E73",
    "blackbox": "#0072B2",
    "residual": "#7E57C2",
    "oracle": "#6B7280",
}
METHOD_MARKERS = {
    "nominal": "o",
    "fitted_physics": "s",
    "blackbox": "^",
    "residual": "D",
    "oracle": "x",
}
BACKGROUND = "#F7F7F3"

BADGE_CONFIRMATORY = "Confirmatory result"
BADGE_DIAGNOSTIC = "Diagnostic"
BADGE_SCHEMATIC = "Schematic explanation"


def _new_figure(width: float = 6.4, height: float = 4.2) -> tuple[Any, Any]:
    figure, axis = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    return figure, axis


def _finish(
    figure: Any,
    output_dir: Path,
    name: str,
    badge: str,
    caption: str,
    displayed_data: Any,
    registry: list[dict[str, Any]],
    source: str,
) -> None:
    figure.text(0.01, 0.01, badge, fontsize=7, color="#6B7280")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    svg = output_dir / f"{name}.svg"
    png = output_dir / f"{name}@2x.png"
    figure.savefig(svg)
    figure.savefig(png, dpi=200)
    plt.close(figure)
    (output_dir / f"{name}.data.json").write_text(
        json.dumps(displayed_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry.append(
        {
            "figure_id": name,
            "badge": badge,
            "caption": caption,
            "source": source,
            "files": [svg.name, png.name, f"{name}.data.json"],
        }
    )


def _placeholder(
    output_dir: Path, name: str, message: str, registry: list[dict[str, Any]]
) -> None:
    figure, axis = _new_figure()
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, wrap=True)
    _finish(
        figure, output_dir, name, BADGE_SCHEMATIC, message, {"message": message},
        registry, "none",
    )


def render_figures(
    contract: ExperimentContract, analysis_directory: Path, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry: list[dict[str, Any]] = []

    contrasts = pq.read_table(analysis_directory / "contrasts.parquet").to_pydict()
    methods_table = pq.read_table(
        analysis_directory / "method_summaries.parquet"
    ).to_pydict()
    cells = pq.read_table(analysis_directory / "cell_outcomes.parquet").to_pydict()
    interpretation = json.loads(
        (analysis_directory / "interpretation.json").read_text(encoding="utf-8")
    )
    with np.load(analysis_directory / "panel.npz") as panel_file:
        panel_values = panel_file["values"]
        panel_methods = [str(m) for m in panel_file["methods"]]
    prediction_path = analysis_directory / "prediction_summaries.parquet"
    predictions = (
        pq.read_table(prediction_path).to_pydict() if prediction_path.exists() else None
    )

    _figure_f03_primary_effect(
        contract, contrasts, panel_values, panel_methods, interpretation, output_dir,
        registry,
    )
    _figure_f04_budget_curve(contract, methods_table, output_dir, registry)
    if predictions is not None:
        _figure_f05_horizon_error(predictions, output_dir, registry)
        _figure_f06_prediction_control(
            contract, predictions, cells, output_dir, registry
        )
    else:
        _placeholder(output_dir, "F05_horizon_error",
                     "Prediction metrics not evaluated under the frozen scope.", registry)
        _placeholder(output_dir, "F06_prediction_control",
                     "Prediction metrics not evaluated under the frozen scope.", registry)
    _figure_f07_world_method_matrix(methods_table, output_dir, registry)
    _figure_f11_failure_taxonomy(methods_table, output_dir, registry)
    _figure_f12_runtime(cells, output_dir, registry)
    _figure_f14_evidence_grid(panel_values, panel_methods, output_dir, registry)

    (output_dir / "figure_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"figures": [entry["figure_id"] for entry in registry],
            "directory": str(output_dir)}


def _figure_f03_primary_effect(
    contract: ExperimentContract,
    contrasts: dict[str, list[Any]],
    panel_values: np.ndarray,
    panel_methods: list[str],
    interpretation: dict[str, Any],
    output_dir: Path,
    registry: list[dict[str, Any]],
) -> None:
    """Pipeline-level paired primary contrast: the central result view."""
    index = contrasts["contrast_id"].index(contract.analysis.primary_contrast.contrast_id)
    estimate = contrasts["estimate"][index]
    lower, upper = contrasts["lower_95"][index], contrasts["upper_95"][index]
    a = panel_values[:, :, panel_methods.index("residual")]
    b = panel_values[:, :, panel_methods.index("blackbox")]
    pipeline_diffs = np.mean(a - b, axis=1)

    figure, axis = _new_figure(6.0, 4.0)
    jitter = np.linspace(-0.08, 0.08, pipeline_diffs.shape[0])
    axis.scatter(
        jitter, pipeline_diffs, color=METHOD_COLORS["residual"], marker="D", zorder=3,
        label="pipeline (scenario-averaged)",
    )
    axis.errorbar(
        [0.35], [estimate], yerr=[[estimate - lower], [upper - estimate]],
        fmt="o", color="#202124", capsize=5, zorder=4, label="estimate (95% crossed CI)",
    )
    axis.axhline(0.0, color="#9A9A94", linewidth=1)
    threshold = contract.analysis.practical_effect_threshold_absolute
    axis.axhline(threshold, color="#9A9A94", linewidth=1, linestyle=":")
    axis.text(0.42, threshold, f"practical threshold +{threshold:.2f}",
              fontsize=7, va="bottom", color="#6B7280")
    axis.set_xlim(-0.3, 0.9)
    axis.set_xticks([])
    axis.set_ylabel("paired success difference (residual - black box)")
    axis.set_title(
        f"Primary contrast, {interpretation['pipelines']} pipelines x "
        f"{interpretation['scenarios']} scenarios"
    )
    axis.legend(fontsize=7, loc="lower right")
    _finish(
        figure, output_dir, "F03_primary_effect", BADGE_CONFIRMATORY,
        "Paired residual-minus-black-box task success with crossed bootstrap interval.",
        {
            "pipeline_differences": pipeline_diffs.tolist(),
            "estimate": estimate, "lower_95": lower, "upper_95": upper,
            "practical_threshold": threshold,
            "interpretation_state": interpretation["state"],
        },
        registry, "contrasts.parquet",
    )


def _figure_f04_budget_curve(
    contract: ExperimentContract,
    methods_table: dict[str, list[Any]],
    output_dir: Path,
    registry: list[dict[str, Any]],
) -> None:
    figure, axis = _new_figure()
    world = contract.evaluation.primary_world
    displayed: dict[str, Any] = {}
    for method in ("fitted_physics", "blackbox", "residual"):
        budgets, rates = [], []
        for i in range(len(methods_table["method"])):
            if methods_table["method"][i] == method and methods_table["world_id"][i] == world:
                budgets.append(methods_table["budget"][i])
                rates.append(methods_table["success_rate"][i])
        order = np.argsort(budgets)
        budgets = [budgets[j] for j in order]
        rates = [rates[j] for j in order]
        axis.plot(
            budgets, rates, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
            label=METHOD_DISPLAY_NAMES[method],
        )
        displayed[method] = {"budgets": budgets, "success_rates": rates}
    for method in ("nominal", "oracle"):
        rates = [
            methods_table["success_rate"][i]
            for i in range(len(methods_table["method"]))
            if methods_table["method"][i] == method and methods_table["world_id"][i] == world
        ]
        if rates:
            axis.axhline(
                float(np.mean(rates)), color=METHOD_COLORS[method], linestyle="--",
                linewidth=1, label=f"{METHOD_DISPLAY_NAMES[method]} (budget-free)",
            )
            displayed[method] = {"success_rate": float(np.mean(rates))}
    axis.set_xscale("log")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("adaptation transitions (train + validation)")
    axis.set_ylabel("task success rate")
    axis.set_title(f"Closed-loop success vs data budget ({world})")
    axis.legend(fontsize=7)
    _finish(
        figure, output_dir, "F04_budget_curve", BADGE_DIAGNOSTIC,
        "Success against adaptation budget; only executed budgets are shown.",
        displayed, registry, "method_summaries.parquet",
    )


def _figure_f05_horizon_error(
    predictions: dict[str, list[Any]], output_dir: Path, registry: list[dict[str, Any]]
) -> None:
    figure, axis = _new_figure()
    displayed: dict[str, Any] = {}
    for method in ("fitted_physics", "blackbox", "residual"):
        horizons: dict[int, list[float]] = {}
        for i in range(len(predictions["method"])):
            if predictions["method"][i] != method or predictions["rmse"][i] is None:
                continue
            horizons.setdefault(predictions["horizon"][i], []).append(
                predictions["rmse"][i]
            )
        keys = sorted(horizons)
        means = [float(np.mean(horizons[h])) for h in keys]
        axis.plot(
            keys, means, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
            label=METHOD_DISPLAY_NAMES[method],
        )
        displayed[method] = {"horizons": keys, "rmse": means}
    axis.set_yscale("log")
    axis.set_xlabel("open-loop rollout horizon (control steps)")
    axis.set_ylabel("normalized state RMSE (finite origins)")
    axis.set_title("Compounding open-loop prediction error")
    axis.legend(fontsize=7)
    _finish(
        figure, output_dir, "F05_horizon_error", BADGE_DIAGNOSTIC,
        "Held-out open-loop error by horizon, conditional on finite origins.",
        displayed, registry, "prediction_summaries.parquet",
    )


def _figure_f06_prediction_control(
    contract: ExperimentContract,
    predictions: dict[str, list[Any]],
    cells: dict[str, list[Any]],
    output_dir: Path,
    registry: list[dict[str, Any]],
) -> None:
    """One point per (pipeline, method, budget): never pseudo-replicated."""
    max_horizon = max(contract.data.rollout_horizons)
    figure, axis = _new_figure()
    displayed: list[dict[str, Any]] = []
    for method in ("fitted_physics", "blackbox", "residual"):
        for i in range(len(predictions["method"])):
            if (
                predictions["method"][i] != method
                or predictions["horizon"][i] != max_horizon
                or predictions["rmse"][i] is None
            ):
                continue
            replicate = predictions["replicate"][i]
            budget = predictions["budget"][i]
            successes = [
                cells["success"][j]
                for j in range(len(cells["method"]))
                if cells["method"][j] == method
                and cells["replicate"][j] == replicate
                and cells["budget"][j] == budget
            ]
            if not successes:
                continue
            success_rate = float(np.mean(successes))
            axis.scatter(
                predictions["rmse"][i], success_rate,
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], alpha=0.85,
            )
            displayed.append(
                {
                    "method": method, "replicate": replicate, "budget": budget,
                    "rmse": predictions["rmse"][i], "success_rate": success_rate,
                }
            )
    axis.set_xscale("log")
    axis.set_xlabel(f"held-out {max_horizon}-step normalized RMSE")
    axis.set_ylabel("scenario-averaged task success")
    axis.set_title("Prediction error vs closed-loop control (association only)")
    handles = [
        Line2D([], [], marker=METHOD_MARKERS[m], linestyle="",
                   color=METHOD_COLORS[m], label=METHOD_DISPLAY_NAMES[m])
        for m in ("fitted_physics", "blackbox", "residual")
    ]
    axis.legend(handles=handles, fontsize=7)
    _finish(
        figure, output_dir, "F06_prediction_control", BADGE_DIAGNOSTIC,
        "Association between held-out prediction error and control success; "
        "one point per pipeline/method/budget, no causal claim.",
        displayed, registry, "prediction_summaries.parquet + cell_outcomes.parquet",
    )


def _figure_f07_world_method_matrix(
    methods_table: dict[str, list[Any]], output_dir: Path, registry: list[dict[str, Any]]
) -> None:
    worlds = sorted(set(methods_table["world_id"]))
    methods = [m for m in METHOD_COLORS if m in set(methods_table["method"])]
    # Pool successes/episodes over budgets; exact per-budget numbers stay
    # in the data file and tables.
    successes = np.zeros((len(methods), len(worlds)))
    counts = np.zeros((len(methods), len(worlds)))
    for i in range(len(methods_table["method"])):
        m = methods.index(methods_table["method"][i])
        w = worlds.index(methods_table["world_id"][i])
        successes[m, w] += methods_table["successes"][i]
        counts[m, w] += methods_table["episodes"][i]
    with np.errstate(invalid="ignore"):
        matrix = np.where(counts > 0, successes / np.maximum(counts, 1), np.nan)
    figure, axis = _new_figure(5.6, 4.0)
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(worlds)), worlds, rotation=30, ha="right", fontsize=7)
    axis.set_yticks(
        range(len(methods)), [METHOD_DISPLAY_NAMES[m] for m in methods], fontsize=8
    )
    for m in range(len(methods)):
        for w in range(len(worlds)):
            if not np.isnan(matrix[m, w]):
                axis.text(
                    w, m, f"{matrix[m, w]:.2f}\nn={int(counts[m, w])}",
                    ha="center", va="center", fontsize=6,
                    color="white" if matrix[m, w] < 0.6 else "black",
                )
    figure.colorbar(image, ax=axis, label="success rate")
    axis.set_title("Success by world and method (pooled over budgets)")
    _finish(
        figure, output_dir, "F07_world_method_matrix", BADGE_DIAGNOSTIC,
        "World-by-method success rates with episode counts.",
        {"worlds": worlds, "methods": methods, "matrix": matrix.tolist(),
         "episodes": counts.tolist()},
        registry, "method_summaries.parquet",
    )


def _figure_f11_failure_taxonomy(
    methods_table: dict[str, list[Any]], output_dir: Path, registry: list[dict[str, Any]]
) -> None:
    methods = sorted(set(methods_table["method"]), key=list(METHOD_COLORS).index)
    categories = ("successes", "collision", "hard_limit", "timeout", "nonfinite",
                  "training_failed")
    labels = ("success", "collision", "hard limit/speed", "timeout", "non-finite",
              "training failed")
    colors = ("#009E73", "#C94F4F", "#A65F00", "#9A9A94", "#4A4A46", "#0072B2")
    totals = {
        method: {
            category: sum(
                methods_table[category][i]
                for i in range(len(methods_table["method"]))
                if methods_table["method"][i] == method
            )
            for category in categories
        }
        for method in methods
    }
    figure, axis = _new_figure()
    bottoms = np.zeros(len(methods))
    for category, label, color in zip(categories, labels, colors, strict=True):
        values = np.array([totals[m][category] for m in methods], dtype=float)
        axis.bar(range(len(methods)), values, bottom=bottoms, label=label, color=color)
        bottoms += values
    axis.set_xticks(
        range(len(methods)), [METHOD_DISPLAY_NAMES[m] for m in methods],
        rotation=20, ha="right", fontsize=8,
    )
    axis.set_ylabel("episodes")
    axis.set_title("Outcome composition (all scheduled episodes)")
    axis.legend(fontsize=7)
    _finish(
        figure, output_dir, "F11_failure_taxonomy", BADGE_DIAGNOSTIC,
        "Every scheduled episode by terminal outcome; denominators are complete.",
        totals, registry, "method_summaries.parquet",
    )


def _figure_f12_runtime(
    cells: dict[str, list[Any]], output_dir: Path, registry: list[dict[str, Any]]
) -> None:
    methods = sorted(
        {m for m in cells["method"]}, key=list(METHOD_COLORS).index
    )
    figure, axis = _new_figure()
    displayed = {}
    for position, method in enumerate(methods):
        p50s = [
            cells["planning_time_ms_p50"][i]
            for i in range(len(cells["method"]))
            if cells["method"][i] == method and cells["planning_time_ms_p50"][i] is not None
        ]
        p95s = [
            cells["planning_time_ms_p95"][i]
            for i in range(len(cells["method"]))
            if cells["method"][i] == method and cells["planning_time_ms_p95"][i] is not None
        ]
        if not p50s:
            continue
        axis.scatter([position] * len(p50s), p50s, color=METHOD_COLORS[method],
                     marker=METHOD_MARKERS[method], s=14, alpha=0.6, label=None)
        axis.scatter([position + 0.2] * len(p95s), p95s, color=METHOD_COLORS[method],
                     marker="_", s=60, alpha=0.6)
        displayed[method] = {"p50_ms": p50s, "p95_ms": p95s}
    axis.axhline(50.0, color="#C94F4F", linewidth=1, linestyle=":")
    axis.text(0.02, 50.0, "50 ms control period", fontsize=7, va="bottom",
              color="#C94F4F")
    axis.set_xticks(
        range(len(methods)), [METHOD_DISPLAY_NAMES[m] for m in methods],
        rotation=20, ha="right", fontsize=8,
    )
    axis.set_yscale("log")
    axis.set_ylabel("per-episode planning time (ms; p50 dots, p95 dashes)")
    axis.set_title("Planning latency by method")
    _finish(
        figure, output_dir, "F12_runtime", BADGE_DIAGNOSTIC,
        "Planning-time distributions; the simulated controller does not miss "
        "deadlines, so this is a deployment diagnostic only.",
        displayed, registry, "cell_outcomes.parquet",
    )


def _figure_f14_evidence_grid(
    panel_values: np.ndarray,
    panel_methods: list[str],
    output_dir: Path,
    registry: list[dict[str, Any]],
) -> None:
    """Paired residual/black-box agreement per pipeline x scenario."""
    residual = panel_values[:, :, panel_methods.index("residual")]
    blackbox = panel_values[:, :, panel_methods.index("blackbox")]
    # 0 both fail, 1 blackbox only, 2 residual only, 3 both succeed.
    categories = residual * 2 + blackbox
    from matplotlib.colors import ListedColormap

    colormap = ListedColormap(["#4A4A46", "#0072B2", "#7E57C2", "#009E73"])
    figure, axis = _new_figure(6.4, 3.6)
    axis.imshow(categories, cmap=colormap, vmin=0, vmax=3, aspect="auto")
    axis.set_xlabel("protected scenario")
    axis.set_ylabel("pipeline replicate")
    axis.set_title("Paired outcome agreement (residual vs black box)")
    legend_handles = [
        Rectangle((0, 0), 1, 1, color=c)
        for c in ("#4A4A46", "#0072B2", "#7E57C2", "#009E73")
    ]
    axis.legend(
        legend_handles,
        ["both fail", "black box only", "residual only", "both succeed"],
        fontsize=6, loc="upper right",
    )
    _finish(
        figure, output_dir, "F14_evidence_grid", BADGE_DIAGNOSTIC,
        "Per-cell paired outcomes; pairing structure is visible, not averaged away.",
        {"categories": categories.tolist()},
        registry, "panel.npz",
    )
