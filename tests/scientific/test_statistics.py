"""Statistics on synthetic crossed arrays with known answers."""

import numpy as np
import pytest

from residual_worlds.analysis.statistics import (
    BootstrapResult,
    PairedPanel,
    crossed_stratified_bootstrap,
    exact_sign_flip_p_value,
    holm_adjust,
    interpretation_state,
)

pytestmark = pytest.mark.scientific

ROOT = 730241


def _panel(values: np.ndarray, strata: tuple[int, ...] | None = None) -> PairedPanel:
    if strata is None:
        strata = tuple(range(values.shape[1]))
    return PairedPanel(values=values, methods=("a", "b"), scenario_strata=strata)


def test_paired_difference_known_value() -> None:
    # 2 pipelines x 4 scenarios; a beats b by exactly 0.25 on average.
    a = np.array([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=float)
    b = np.array([[1, 0, 0, 1], [0, 0, 1, 1]], dtype=float)
    panel = _panel(np.stack([a, b], axis=-1))
    assert panel.paired_difference("a", "b") == pytest.approx(0.25)
    np.testing.assert_allclose(panel.pipeline_differences("a", "b"), [0.25, 0.25])


def test_bootstrap_zero_variance_gives_zero_width_interval() -> None:
    # Constant difference everywhere: every resample reproduces it.
    a = np.ones((4, 6))
    b = np.full((4, 6), 0.5)
    panel = _panel(np.stack([a, b], axis=-1), strata=(0, 0, 1, 1, 2, 2))
    result = crossed_stratified_bootstrap(panel, "a", "b", 500, ROOT, "const")
    assert result.estimate == pytest.approx(0.5)
    assert result.lower == pytest.approx(0.5)
    assert result.upper == pytest.approx(0.5)


def test_bootstrap_interval_brackets_estimate_and_is_seeded() -> None:
    rng = np.random.Generator(np.random.PCG64DXSM(3))
    a = rng.uniform(0, 1, size=(8, 12))
    b = a - 0.15 + rng.normal(0, 0.05, size=(8, 12))
    panel = _panel(np.stack([a, b], axis=-1), strata=tuple(i // 2 for i in range(12)))
    first = crossed_stratified_bootstrap(panel, "a", "b", 800, ROOT, "seeded")
    second = crossed_stratified_bootstrap(panel, "a", "b", 800, ROOT, "seeded")
    np.testing.assert_array_equal(first.draws, second.draws)
    assert first.lower <= first.estimate <= first.upper
    assert first.lower > 0.0  # a clear 0.15 advantage resolves above zero


def test_bootstrap_preserves_stratum_composition() -> None:
    # Make one stratum carry a huge effect and the other zero. With the
    # per-stratum resampling, every draw contains exactly half of each,
    # so draws concentrate near the pooled mean rather than mixing
    # stratum proportions.
    a = np.zeros((6, 4))
    a[:, :2] = 1.0  # stratum 0 columns
    b = np.zeros((6, 4))
    panel = _panel(np.stack([a, b], axis=-1), strata=(0, 0, 1, 1))
    result = crossed_stratified_bootstrap(panel, "a", "b", 400, ROOT, "strata")
    # Pipelines and within-stratum columns are exchangeable here, so all
    # draws equal exactly 0.5.
    assert result.lower == pytest.approx(0.5)
    assert result.upper == pytest.approx(0.5)


def test_exact_sign_flip_enumeration_known_values() -> None:
    # All positive, R = 4: only the all-positive and all-negative
    # assignments reach |mean| >= observed -> p = 2 / 16.
    assert exact_sign_flip_p_value(np.array([0.2, 0.2, 0.2, 0.2])) == pytest.approx(2 / 16)
    # A zero vector: every assignment ties -> p = 1.
    assert exact_sign_flip_p_value(np.zeros(4)) == pytest.approx(1.0)


def test_holm_adjustment_known_case() -> None:
    adjusted = holm_adjust({"x": 0.01, "y": 0.04})
    assert adjusted["x"] == pytest.approx(0.02)
    assert adjusted["y"] == pytest.approx(0.04)
    # Monotonicity enforcement.
    adjusted = holm_adjust({"x": 0.03, "y": 0.031})
    assert adjusted["y"] >= adjusted["x"]


def _result(estimate: float, lower: float, upper: float) -> BootstrapResult:
    return BootstrapResult(
        estimate=estimate, lower=lower, upper=upper, draws=np.zeros(1), replicates=1
    )


def test_interpretation_states_follow_frozen_rule() -> None:
    threshold = 0.10
    assert (
        interpretation_state(_result(0.15, 0.02, 0.30), threshold)
        == "supports_primary_direction"
    )
    # Positive but below the practical threshold: not supported.
    assert (
        interpretation_state(_result(0.06, 0.01, 0.12), threshold)
        == "small_or_inconclusive"
    )
    # Interval crossing zero: not supported even with a big estimate.
    assert (
        interpretation_state(_result(0.20, -0.05, 0.40), threshold)
        == "small_or_inconclusive"
    )
    assert (
        interpretation_state(_result(-0.2, -0.4, -0.05), threshold) == "opposite_direction"
    )
    assert interpretation_state(_result(0.0, 0.0, 0.0), threshold, any_rows=False) == (
        "no_results"
    )
