"""Training runs: matched neural members and the fitted-physics baseline.

Fairness is enforced structurally, not by good intentions:

* black-box and residual member ``e`` draw their collection-unit
  bootstrap multiplicities and their entire ordered minibatch sequence
  from method-free ``neural_pair`` seed namespaces, so both methods see
  identical data exposure update by update;
* only the initialization namespace is method-qualified, because the
  residual head must start at zero while the black box uses the seeded
  default rule -- that asymmetry is the tested prior;
* every member trains for the same fixed update count regardless of
  budget, with the same optimizer schedule, and selects the earliest
  checkpoint minimizing the mean normalized five-step validation loss;
* the fitted-physics baseline consumes the same train/validation rows
  through the same one-step-plus-five-step objective.

A run that diverges numerically produces a structurally complete
``TRAINING_FAILED`` artifact -- a scientific outcome, never a silent
retry.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file

from residual_worlds.config import ExperimentContract
from residual_worlds.data.dataset import DatasetView, load_dataset
from residual_worlds.identity import content_id
from residual_worlds.models.base import Stepper
from residual_worlds.models.black_box import BlackBoxModel
from residual_worlds.models.fitted_physics import (
    PARAMETER_ORDER,
    FitResult,
    fit_fitted_physics,
    fitted_acceleration,
)
from residual_worlds.models.normalization import PhysicalScales
from residual_worlds.models.residual import ResidualModel
from residual_worlds.paths import run_dir
from residual_worlds.provenance import is_complete, verify_artifact, write_artifact
from residual_worlds.seeds import numpy_generator, seed_record, torch_generator
from residual_worlds.training.losses import combined_loss, normalized_state_loss

NEURAL_METHODS = ("blackbox", "residual")


def run_spec(
    contract: ExperimentContract,
    method: str,
    world_id: str,
    budget: int,
    replicate: int,
    member: int,
    dataset_id: str,
) -> dict[str, Any]:
    """Precomputable run specification (its hash is the run id)."""
    return {
        "schema": 1,
        "method": method,
        "world_id": world_id,
        "budget": budget,
        "replicate": replicate,
        "member": member,
        "dataset_id": dataset_id,
        "root_seed": contract.numerics.root_seed,
        "updates": contract.training.updates,
        "architecture": {
            "hidden_widths": list(contract.models.neural_common.hidden_widths),
            "activation": contract.models.neural_common.activation,
        },
        "one_step_batch": contract.training.one_step_batch_size,
        "rollout_batch": contract.training.rollout_batch_size,
        "rollout_horizon": contract.training.rollout_horizon,
        "learning_rate": [
            contract.training.learning_rate_initial,
            contract.training.learning_rate_final,
        ],
        "weight_decay": contract.training.weight_decay,
    }


@dataclass(frozen=True)
class _BudgetData:
    train_rows: np.ndarray
    validation_rows: np.ndarray
    train_membership_digest: str
    validation_membership_digest: str


def _budget_data(
    contract: ExperimentContract, view: DatasetView, budget: int
) -> _BudgetData:
    unit = contract.data.collection_unit_valid_transitions
    fraction = contract.data.train_fraction
    train_units, validation_units = view.units_for_budget(budget, unit, fraction)
    return _BudgetData(
        train_rows=view.rows_for_units(train_units),
        validation_rows=view.rows_for_units(validation_units),
        train_membership_digest=view.membership_digest(train_units),
        validation_membership_digest=view.membership_digest(validation_units),
    )


def _bootstrap_rows(
    contract: ExperimentContract,
    view: DatasetView,
    budget: int,
    replicate: int,
    member: int,
    world_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrapped training rows (by whole units) and their multiplicities.

    The namespace contains no method token: residual and black-box
    member ``e`` receive the same multiplicities by construction.
    """
    unit = contract.data.collection_unit_valid_transitions
    fraction = contract.data.train_fraction
    train_units, _ = view.units_for_budget(budget, unit, fraction)
    rng = numpy_generator(
        contract.numerics.root_seed, "bootstrap", "neural_pair", world_id, budget,
        replicate, member,
    )
    draws = rng.integers(0, len(train_units), size=len(train_units))
    multiplicities = np.bincount(draws, minlength=len(train_units))
    rows: list[np.ndarray] = []
    for index, count in enumerate(multiplicities):
        if count == 0:
            continue
        unit_rows = view.rows_for_units((train_units[index],))
        rows.extend([unit_rows] * int(count))
    return np.concatenate(rows), multiplicities


def _build_member(
    contract: ExperimentContract, method: str, replicate: int, member: int
) -> torch.nn.Module:
    scales = PhysicalScales.from_contract(contract)
    generator = torch_generator(
        contract.numerics.root_seed, "model_init", method, replicate, member
    )
    if method == "blackbox":
        return BlackBoxModel(contract.models.neural_common, scales, generator)
    if method == "residual":
        return ResidualModel(contract.models.neural_common, scales, contract.arm, generator)
    raise ValueError(f"unknown neural method {method!r}")


def _segments(
    view: DatasetView, rows: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(initial states, action sequences, truth states) for all eligible origins."""
    origins = view.rollout_origins(rows, horizon)
    if origins.size == 0:
        raise RuntimeError("no eligible rollout segments in the given rows")
    initial = view.state[origins]
    actions = np.stack([view.action[origins + h] for h in range(horizon)], axis=1)
    truth = np.stack([view.next_state[origins + h] for h in range(horizon)], axis=1)
    return initial, actions, truth


def train_neural_member(
    contract: ExperimentContract,
    dataset_dir: Path,
    method: str,
    world_id: str,
    budget: int,
    replicate: int,
    member: int,
    device: str = "cpu",
) -> dict[str, Any]:
    if method not in NEURAL_METHODS:
        raise ValueError(f"{method!r} is not a neural method")
    view = load_dataset(dataset_dir)
    dataset_id = dataset_dir.name
    spec = run_spec(contract, method, world_id, budget, replicate, member, dataset_id)
    run_id = content_id("run", spec)
    destination = run_dir(run_id)
    if is_complete(destination):
        verify_artifact(destination)
        stored = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
        return {
            "run_id": run_id,
            "artifact": str(destination),
            "reused": True,
            "status": stored["status"],
            "selected_update": stored["selected_update"],
            "selected_validation_loss": stored["selected_validation_loss"],
        }

    training = contract.training
    stepper = Stepper.from_contract(contract)
    dtype = torch.float32
    budget_rows = _budget_data(contract, view, budget)
    boot_rows, multiplicities = _bootstrap_rows(
        contract, view, budget, replicate, member, world_id
    )
    horizon = training.rollout_horizon
    boot_origins = view.rollout_origins(np.unique(boot_rows), horizon)
    validation_initial, validation_actions, validation_truth = _segments(
        view, budget_rows.validation_rows, horizon
    )

    model = _build_member(contract, method, replicate, member).to(device=device, dtype=dtype)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate_initial,
        weight_decay=training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training.updates, eta_min=training.learning_rate_final
    )
    minibatch_rng = numpy_generator(
        contract.numerics.root_seed, "minibatch", "neural_pair", world_id, budget,
        replicate, member,
    )

    def accel(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = model.acceleration(state, action)  # type: ignore[operator]
        return result

    def to_device(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device=device, dtype=dtype)

    validation_tensors = (
        to_device(validation_initial),
        to_device(validation_actions),
        to_device(validation_truth),
    )

    def validation_loss() -> float:
        model.eval()
        with torch.no_grad():
            initial, actions, truth = validation_tensors
            state = initial
            losses = []
            for h in range(horizon):
                state = stepper.step(accel, state, actions[:, h])
                losses.append(normalized_state_loss(state, truth[:, h]))
            value = float(torch.stack(losses).mean())
        model.train()
        return value

    history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_update = -1
    best_state: dict[str, torch.Tensor] | None = None
    failed_reason: str | None = None
    started = time.perf_counter()

    for update in range(1, training.updates + 1):
        # Ordered method-free draws: one-step rows, then segment origins.
        row_draw = minibatch_rng.integers(0, boot_rows.shape[0], training.one_step_batch_size)
        one_step_rows = boot_rows[row_draw]
        if boot_origins.size > 0:
            origin_draw = minibatch_rng.integers(
                0, boot_origins.shape[0], training.rollout_batch_size
            )
            origins = boot_origins[origin_draw]
            rollout_batch = (
                to_device(view.state[origins]),
                to_device(
                    np.stack([view.action[origins + h] for h in range(horizon)], axis=1)
                ),
                to_device(
                    np.stack([view.next_state[origins + h] for h in range(horizon)], axis=1)
                ),
            )
        else:
            rollout_batch = None

        one_step_batch = (
            to_device(view.state[one_step_rows]),
            to_device(view.action[one_step_rows]),
            to_device(view.next_state[one_step_rows]),
        )
        optimizer.zero_grad()
        loss = combined_loss(
            accel,
            stepper,
            one_step_batch,
            rollout_batch,
            training.one_step_weight,
            training.five_step_weight,
        )
        if not torch.isfinite(loss):
            failed_reason = f"non-finite training loss at update {update}"
            break
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_norm_clip)
        optimizer.step()
        scheduler.step()

        if update % training.validation_every_updates == 0 or update == training.updates:
            value = validation_loss()
            history.append(
                {
                    "update": update,
                    "train_loss": float(loss.detach()),
                    "validation_loss": value,
                }
            )
            if not math.isfinite(value):
                failed_reason = f"non-finite validation loss at update {update}"
                break
            # Earliest minimum: strictly-smaller only.
            if value < best_loss:
                best_loss = value
                best_update = update
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }

    wall_time = time.perf_counter() - started
    scales = PhysicalScales.from_contract(contract)
    metadata = {
        "run_id": run_id,
        "spec": spec,
        "parameter_count": parameter_count,
        "train_membership_sha256": budget_rows.train_membership_digest,
        "validation_membership_sha256": budget_rows.validation_membership_digest,
        "bootstrap_multiplicities": [int(m) for m in multiplicities],
        "bootstrap_seed": seed_record(
            contract.numerics.root_seed, "bootstrap", "neural_pair", world_id, budget,
            replicate, member,
        ).digest_sha256,
        "minibatch_seed": seed_record(
            contract.numerics.root_seed, "minibatch", "neural_pair", world_id, budget,
            replicate, member,
        ).digest_sha256,
        "init_seed": seed_record(
            contract.numerics.root_seed, "model_init", method, replicate, member
        ).digest_sha256,
        "selected_update": best_update,
        "selected_validation_loss": best_loss if math.isfinite(best_loss) else None,
        "wall_time_s": wall_time,
        "device": device,
        "status": "TRAINING_FAILED" if failed_reason else "READY",
        "failure_reason": failed_reason,
    }

    def populate(directory: Path) -> None:
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (directory / "normalizer.json").write_text(
            json.dumps(scales.to_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if history:
            pq.write_table(
                pa.table(
                    {
                        "update": [h["update"] for h in history],
                        "train_loss": [h["train_loss"] for h in history],
                        "validation_loss": [h["validation_loss"] for h in history],
                    }
                ),
                directory / "history.parquet",
            )
        if failed_reason is None and best_state is not None:
            save_file(
                {k: v.cpu() for k, v in best_state.items()},
                str(directory / "model.safetensors"),
            )
            # Stored validation predictions at the selected checkpoint,
            # for reload-reproducibility audits.
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                predicted = stepper.step(
                    accel,
                    to_device(view.state[budget_rows.validation_rows]),
                    to_device(view.action[budget_rows.validation_rows]),
                )
            np.savez(
                directory / "validation_predictions.npz",
                rows=budget_rows.validation_rows,
                predicted_next_state=predicted.cpu().numpy(),
            )
        else:
            (directory / "failure.json").write_text(
                json.dumps({"reason": failed_reason}, indent=2) + "\n", encoding="utf-8"
            )

    write_artifact(
        destination,
        "training_run",
        {"run_id": run_id, "method": method, "member": member},
        {"dataset": dataset_id, "contract": contract.source_path},
        populate,
    )
    return {
        "run_id": run_id,
        "artifact": str(destination),
        "reused": False,
        "status": metadata["status"],
        "selected_update": best_update,
        "selected_validation_loss": metadata["selected_validation_loss"],
    }


def load_trained_member(
    contract: ExperimentContract, run_directory: Path, device: str = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct a trained member from its immutable run artifact."""
    verify_artifact(run_directory)
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "READY":
        raise RuntimeError(f"run {run_directory.name} is {metadata['status']}")
    spec = metadata["spec"]
    model = _build_member(contract, spec["method"], spec["replicate"], spec["member"])
    state = load_file(str(run_directory / "model.safetensors"))
    model.load_state_dict(state)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, metadata


def fit_physics_run(
    contract: ExperimentContract,
    dataset_dir: Path,
    world_id: str,
    budget: int,
    replicate: int,
) -> dict[str, Any]:
    """Fitted-physics baseline on the same rows and objective."""
    view = load_dataset(dataset_dir)
    dataset_id = dataset_dir.name
    spec = run_spec(
        contract, "fitted_physics", world_id, budget, replicate, 0, dataset_id
    )
    run_id = content_id("run", spec)
    destination = run_dir(run_id)
    if is_complete(destination):
        verify_artifact(destination)
        stored = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
        return {
            "run_id": run_id,
            "artifact": str(destination),
            "reused": True,
            "status": stored["status"],
            "theta": list(load_fitted_run(destination)),
            "boundary_hits": stored["boundary_hits"],
        }

    stepper = Stepper.from_contract(contract)
    training = contract.training
    budget_rows = _budget_data(contract, view, budget)
    horizon = training.rollout_horizon
    train_one = view.tensors(budget_rows.train_rows, torch.float64)
    validation_one = view.tensors(budget_rows.validation_rows, torch.float64)
    train_segments = _segments(view, budget_rows.train_rows, horizon)
    validation_segments = _segments(view, budget_rows.validation_rows, horizon)

    def make_loss(
        one: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        segments: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        initial = torch.from_numpy(segments[0])
        actions = torch.from_numpy(segments[1])
        truth = torch.from_numpy(segments[2])

        def loss(theta: torch.Tensor) -> torch.Tensor:
            def accel(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                return fitted_acceleration(state, action, theta, contract.arm)

            return combined_loss(
                accel,
                stepper,
                one,
                (initial, actions, truth),
                training.one_step_weight,
                training.five_step_weight,
            )

        return loss

    started = time.perf_counter()
    result: FitResult = fit_fitted_physics(
        contract,
        make_loss(train_one, train_segments),
        make_loss(validation_one, validation_segments),
        world_id,
        budget,
        replicate,
    )
    wall_time = time.perf_counter() - started

    metadata = {
        "run_id": run_id,
        "spec": spec,
        "parameter_count": len(PARAMETER_ORDER),
        "train_membership_sha256": budget_rows.train_membership_digest,
        "validation_membership_sha256": budget_rows.validation_membership_digest,
        "selected_restart": result.selected_restart,
        "boundary_hits": list(result.boundary_hits),
        "wall_time_s": wall_time,
        "status": "READY",
        "failure_reason": None,
    }

    def populate(directory: Path) -> None:
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (directory / "parameters.json").write_text(
            json.dumps(
                dict(zip(PARAMETER_ORDER, result.theta, strict=True)), indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        pq.write_table(
            pa.table(
                {
                    "restart_index": [r.restart_index for r in result.restarts],
                    "initial_theta": [list(r.initial_theta) for r in result.restarts],
                    "final_theta": [list(r.final_theta) for r in result.restarts],
                    "train_loss": [r.train_loss for r in result.restarts],
                    "validation_loss": [r.validation_loss for r in result.restarts],
                    "converged": [r.converged for r in result.restarts],
                    "iterations": [r.iterations for r in result.restarts],
                }
            ),
            directory / "fit_restarts.parquet",
        )

    write_artifact(
        destination,
        "fitted_physics_run",
        {"run_id": run_id, "method": "fitted_physics"},
        {"dataset": dataset_id, "contract": contract.source_path},
        populate,
    )
    return {
        "run_id": run_id,
        "artifact": str(destination),
        "reused": False,
        "status": "READY",
        "theta": list(result.theta),
        "boundary_hits": list(result.boundary_hits),
    }


def load_fitted_run(run_directory: Path) -> tuple[float, float, float, float, float]:
    verify_artifact(run_directory)
    parameters = json.loads(
        (run_directory / "parameters.json").read_text(encoding="utf-8")
    )
    values = tuple(float(parameters[name]) for name in PARAMETER_ORDER)
    assert len(values) == 5
    return (values[0], values[1], values[2], values[3], values[4])
