"""Model conditions: the immutable method/world/budget/pipeline cells.

The evaluator never globs for checkpoints. For every cell one condition
artifact binds the exact ordered member runs (three for neural methods,
one for fitted physics, none for nominal and the exact-dynamics
reference), and evaluation rows name the condition. A neural member
that failed training fails the whole condition -- the ensemble never
silently shrinks.

The ``oracle`` condition deliberately stores no dynamics: the
evaluation harness, which owns target-world construction, supplies the
exact acceleration at run time. This module cannot import target
physics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from residual_worlds.config import ExperimentContract
from residual_worlds.identity import content_id
from residual_worlds.models.fitted_physics import FittedPhysicsModel
from residual_worlds.models.nominal import NominalModel
from residual_worlds.paths import condition_dir, run_dir
from residual_worlds.physics.integrators import AccelerationFn
from residual_worlds.provenance import is_complete, verify_artifact, write_artifact
from residual_worlds.training.train import (
    load_fitted_run,
    load_trained_member,
    run_spec,
)

ADAPTED_METHODS = ("fitted_physics", "blackbox", "residual")
UNADAPTED_METHODS = ("nominal", "oracle")


def condition_spec(
    contract: ExperimentContract,
    method: str,
    world_id: str,
    budget: int,
    replicate: int,
    dataset_id: str | None,
) -> dict[str, Any]:
    members: list[str] = []
    if method in ("blackbox", "residual"):
        assert dataset_id is not None
        members = [
            content_id(
                "run",
                run_spec(contract, method, world_id, budget, replicate, member, dataset_id),
            )
            for member in range(contract.models.neural_common.ensemble_members)
        ]
    elif method == "fitted_physics":
        assert dataset_id is not None
        members = [
            content_id(
                "run",
                run_spec(contract, method, world_id, budget, replicate, 0, dataset_id),
            )
        ]
    return {
        "schema": 1,
        "method": method,
        "world_id": world_id,
        "budget": budget,
        "replicate": replicate,
        "member_run_ids": members,
        "root_seed": contract.numerics.root_seed,
    }


def build_condition(
    contract: ExperimentContract,
    method: str,
    world_id: str,
    budget: int,
    replicate: int,
    dataset_id: str | None,
) -> dict[str, Any]:
    spec = condition_spec(contract, method, world_id, budget, replicate, dataset_id)
    condition_id = content_id("condition", spec)
    destination = condition_dir(condition_id)
    if is_complete(destination):
        verify_artifact(destination)
        payload = json.loads((destination / "condition.json").read_text(encoding="utf-8"))
        return {
            "condition_id": condition_id,
            "artifact": str(destination),
            "status": payload["status"],
            "reused": True,
        }

    status = "READY"
    failure: str | None = None
    for member_run in spec["member_run_ids"]:
        member_dir = run_dir(member_run)
        if not is_complete(member_dir):
            raise FileNotFoundError(f"member run {member_run} has no complete artifact")
        metadata = json.loads((member_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "READY":
            status = "TRAINING_FAILED"
            failure = f"member {member_run}: {metadata.get('failure_reason')}"
            break

    payload = {
        "condition_id": condition_id,
        "spec": spec,
        "status": status,
        "failure_reason": failure,
    }

    def populate(directory: Path) -> None:
        (directory / "condition.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    write_artifact(
        destination,
        "model_condition",
        {"condition_id": condition_id, "method": method},
        {"member_runs": spec["member_run_ids"]},
        populate,
    )
    return {
        "condition_id": condition_id,
        "artifact": str(destination),
        "status": status,
        "reused": False,
    }


def load_condition_members(
    contract: ExperimentContract,
    condition_directory: Path,
    device: str = "cpu",
) -> tuple[str, list[AccelerationFn]]:
    """(method, member acceleration functions) for a READY condition.

    ``oracle`` returns an empty member list -- the evaluation harness
    supplies the exact target dynamics itself.
    """
    verify_artifact(condition_directory)
    payload = json.loads(
        (condition_directory / "condition.json").read_text(encoding="utf-8")
    )
    if payload["status"] != "READY":
        raise RuntimeError(f"condition is {payload['status']}")
    method = payload["spec"]["method"]
    if method == "nominal":
        model = NominalModel(contract.arm)
        return method, [model.acceleration]
    if method == "oracle":
        return method, []
    if method == "fitted_physics":
        theta = load_fitted_run(run_dir(payload["spec"]["member_run_ids"][0]))
        fitted = FittedPhysicsModel(contract.arm, theta)
        return method, [fitted.acceleration]
    members: list[AccelerationFn] = []
    for member_run in payload["spec"]["member_run_ids"]:
        network, _metadata = load_trained_member(contract, run_dir(member_run), device)

        def acceleration(
            state: object, action: object, _network: object = network
        ) -> object:
            return _network.acceleration(state, action)  # type: ignore[attr-defined]

        members.append(acceleration)  # type: ignore[arg-type]
    return method, members
