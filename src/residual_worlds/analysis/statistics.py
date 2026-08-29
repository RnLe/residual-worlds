"""Paired estimands, crossed stratified bootstrap, and sign-flip checks.

The replication structure is explicit throughout: pipelines and
scenario families are crossed factors, and the R x S episode rows are
never treated as R*S independent experiments. The crossed bootstrap
resamples both axes -- pipelines with replacement, and scenarios with
replacement *within each structural stratum*, preserving the designed
per-stratum composition -- while method pairing inside every selected
cell is preserved exactly.

The exact sign-flip enumeration is an assumption-dependent sensitivity
check (it requires sign-symmetry/exchangeability of the paired pipeline
effects, which method labels were never randomized to guarantee); it
never replaces the interval-plus-practical-threshold rule.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from residual_worlds.seeds import numpy_generator


@dataclass(frozen=True)
class PairedPanel:
    """Success (or metric) values on one crossed panel.

    ``values`` has shape [R, S, M] over pipelines x scenarios x methods;
    ``scenario_strata`` assigns each scenario column a stratum id.
    """

    values: np.ndarray
    methods: tuple[str, ...]
    scenario_strata: tuple[int, ...]

    def method_index(self, method: str) -> int:
        return self.methods.index(method)

    def paired_difference(self, method_a: str, method_b: str) -> float:
        """Mean paired difference a - b with equal cell weights."""
        a = self.values[:, :, self.method_index(method_a)]
        b = self.values[:, :, self.method_index(method_b)]
        return float(np.mean(a - b))

    def pipeline_differences(self, method_a: str, method_b: str) -> np.ndarray:
        """Scenario-averaged paired difference per pipeline, shape [R]."""
        a = self.values[:, :, self.method_index(method_a)]
        b = self.values[:, :, self.method_index(method_b)]
        return np.mean(a - b, axis=1)


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    draws: np.ndarray
    replicates: int


def crossed_stratified_bootstrap(
    panel: PairedPanel,
    method_a: str,
    method_b: str,
    replicates: int,
    root_seed: int,
    contrast_id: str,
    interval: float = 0.95,
) -> BootstrapResult:
    """Percentile interval for the paired contrast a - b.

    Pipelines are resampled with replacement; within each scenario
    stratum, exactly as many scenario columns are resampled (with
    replacement) as the stratum contributes to the panel, so the
    designed composition is preserved in every draw.
    """
    rng = numpy_generator(
        root_seed, "statistics", "crossed_bootstrap", contrast_id
    )
    r_count, s_count, _ = panel.values.shape
    strata = np.asarray(panel.scenario_strata)
    stratum_columns = {
        stratum: np.nonzero(strata == stratum)[0] for stratum in np.unique(strata)
    }
    a = panel.values[:, :, panel.method_index(method_a)]
    b = panel.values[:, :, panel.method_index(method_b)]
    difference = a - b

    draws = np.empty(replicates, dtype=np.float64)
    for draw_index in range(replicates):
        pipelines = rng.integers(0, r_count, size=r_count)
        scenario_columns: list[np.ndarray] = []
        for stratum in sorted(stratum_columns):
            columns = stratum_columns[stratum]
            picked = rng.integers(0, columns.shape[0], size=columns.shape[0])
            scenario_columns.append(columns[picked])
        scenarios = np.concatenate(scenario_columns)
        draws[draw_index] = float(np.mean(difference[pipelines][:, scenarios]))

    alpha = (1.0 - interval) / 2.0
    return BootstrapResult(
        estimate=float(np.mean(difference)),
        lower=float(np.quantile(draws, alpha)),
        upper=float(np.quantile(draws, 1.0 - alpha)),
        draws=draws,
        replicates=replicates,
    )


def exact_sign_flip_p_value(pipeline_differences: np.ndarray) -> float:
    """Two-sided exact sign-flip p-value over all 2^R assignments."""
    r_count = pipeline_differences.shape[0]
    if r_count > 20:
        raise ValueError("exact enumeration is intended for small R")
    observed = abs(float(np.mean(pipeline_differences)))
    at_least_as_extreme = 0
    total = 2**r_count
    for signs in itertools.product((1.0, -1.0), repeat=r_count):
        flipped = float(np.mean(pipeline_differences * np.asarray(signs)))
        if abs(flipped) >= observed - 1e-15:
            at_least_as_extreme += 1
    return float(at_least_as_extreme / total)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment (family = the given dict)."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(ordered):
        value = min(1.0, (m - rank) * p)
        running_max = max(running_max, value)
        adjusted[name] = running_max
    return adjusted


def interpretation_state(
    result: BootstrapResult, practical_threshold: float, any_rows: bool = True
) -> str:
    """Templated interpretation of the primary contrast (frozen rule)."""
    if not any_rows:
        return "no_results"
    if result.lower > 0.0 and result.estimate >= practical_threshold:
        return "supports_primary_direction"
    if result.upper < 0.0:
        return "opposite_direction"
    return "small_or_inconclusive"
