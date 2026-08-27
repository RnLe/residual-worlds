"""Shared test configuration and expensive session fixtures."""

import os
from pathlib import Path

import pytest

from residual_worlds.runtime import configure_torch_cpu

configure_torch_cpu()


@pytest.fixture(scope="session")
def smoke_workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Scenario banks, one dataset, and one prediction set (smoke config).

    Generated once per test session into an isolated artifact root so
    tests never touch the repository's real ``artifacts/`` tree.
    """
    from residual_worlds.config import load_contract
    from residual_worlds.paths import repository_root

    root = tmp_path_factory.mktemp("rw_artifacts")
    scenario_dir = tmp_path_factory.mktemp("rw_scenarios")
    os.environ["RW_ARTIFACT_ROOT"] = str(root)

    contract = load_contract(repository_root() / "configs" / "smoke.yaml")
    from residual_worlds.data.generate import (
        generate_prediction_set,
        generate_world_dataset,
    )
    from residual_worlds.task.scenarios import generate_bank, write_bank_manifest

    for bank in ("training_task", "pilot", "protected"):
        write_bank_manifest(contract, bank, generate_bank(contract, bank), scenario_dir)
    dataset = generate_world_dataset(contract, "composite_standard", 0, scenario_dir)
    prediction = generate_prediction_set(contract, "composite_standard", 0, scenario_dir)
    return {
        "artifact_root": root,
        "scenario_dir": scenario_dir,
        "dataset": Path(dataset["artifact"]),
        "prediction": Path(prediction["artifact"]),
    }
