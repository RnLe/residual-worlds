"""Training runs: pairing, determinism, checkpoint reload, fitted baseline."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from residual_worlds.config import load_contract
from residual_worlds.models.base import Stepper
from residual_worlds.paths import repository_root
from residual_worlds.training.train import (
    fit_physics_run,
    load_fitted_run,
    load_trained_member,
    train_neural_member,
)

pytestmark = [pytest.mark.scientific, pytest.mark.slow]

SMOKE = load_contract(repository_root() / "configs" / "smoke.yaml")
WORLD = "composite_standard"
BUDGET = 128


@pytest.fixture(scope="module")
def trained(smoke_workspace: dict[str, Path]) -> dict[str, dict]:
    results = {}
    for method in ("blackbox", "residual"):
        results[method] = train_neural_member(
            SMOKE, smoke_workspace["dataset"], method, WORLD, BUDGET, 0, 0
        )
    results["fitted_physics"] = fit_physics_run(
        SMOKE, smoke_workspace["dataset"], WORLD, BUDGET, 0
    )
    return results


def test_both_methods_train_to_ready(trained: dict[str, dict]) -> None:
    assert trained["blackbox"]["status"] == "READY"
    assert trained["residual"]["status"] == "READY"
    assert trained["fitted_physics"]["status"] == "READY"


def test_paired_members_share_data_exposure(trained: dict[str, dict]) -> None:
    # Same dataset membership and identical method-free bootstrap and
    # minibatch stream digests: only initialization may differ.
    metas = {}
    for method in ("blackbox", "residual"):
        run = Path(trained[method]["artifact"])
        metas[method] = json.loads((run / "metadata.json").read_text())
    a, b = metas["blackbox"], metas["residual"]
    assert a["train_membership_sha256"] == b["train_membership_sha256"]
    assert a["validation_membership_sha256"] == b["validation_membership_sha256"]
    assert a["bootstrap_multiplicities"] == b["bootstrap_multiplicities"]
    assert a["bootstrap_seed"] == b["bootstrap_seed"]
    assert a["minibatch_seed"] == b["minibatch_seed"]
    assert a["init_seed"] != b["init_seed"]
    assert a["parameter_count"] == b["parameter_count"]


def test_training_reduces_validation_loss(
    trained: dict[str, dict], smoke_workspace: dict[str, Path]
) -> None:
    import pyarrow.parquet as pq

    for method in ("blackbox", "residual"):
        history = pq.read_table(
            Path(trained[method]["artifact"]) / "history.parquet"
        ).to_pydict()
        assert min(history["validation_loss"]) <= history["validation_loss"][0]


def test_checkpoint_reload_reproduces_validation_predictions(
    trained: dict[str, dict], smoke_workspace: dict[str, Path]
) -> None:
    from residual_worlds.data.dataset import load_dataset

    view = load_dataset(smoke_workspace["dataset"])
    stepper = Stepper.from_contract(SMOKE)
    for method in ("blackbox", "residual"):
        run = Path(trained[method]["artifact"])
        model, _meta = load_trained_member(SMOKE, run)
        stored = np.load(run / "validation_predictions.npz")
        rows = stored["rows"]
        with torch.no_grad():
            predicted = stepper.step(
                model.acceleration,
                torch.from_numpy(view.state[rows]).to(torch.float32),
                torch.from_numpy(view.action[rows]).to(torch.float32),
            )
        np.testing.assert_allclose(
            predicted.numpy(), stored["predicted_next_state"], rtol=1e-6, atol=1e-6
        )


def test_rerun_reuses_immutable_artifact(
    trained: dict[str, dict], smoke_workspace: dict[str, Path]
) -> None:
    again = train_neural_member(
        SMOKE, smoke_workspace["dataset"], "residual", WORLD, BUDGET, 0, 0
    )
    assert again["reused"] is True
    assert again["run_id"] == trained["residual"]["run_id"]


def test_fitted_physics_finds_plausible_composite_parameters(
    trained: dict[str, dict],
) -> None:
    # The composite world hides payload 0.25 kg and gains (0.86, 1.12);
    # the five-parameter family cannot represent Stribeck friction or
    # dead zones, but its payload estimate should land near the truth.
    theta = load_fitted_run(Path(trained["fitted_physics"]["artifact"]))
    assert 0.1 <= theta[0] <= 0.45  # payload
    assert 0.5 <= theta[3] <= 1.5 and 0.5 <= theta[4] <= 1.5


def test_no_target_parameters_in_model_visible_files(
    trained: dict[str, dict],
) -> None:
    # Model-loader-visible payloads must not mention hidden world values.
    for method in ("blackbox", "residual"):
        run = Path(trained[method]["artifact"])
        for name in ("metadata.json", "normalizer.json"):
            text = (run / name).read_text()
            assert "payload" not in text
            assert "deadzone" not in text
            assert "coulomb" not in text
