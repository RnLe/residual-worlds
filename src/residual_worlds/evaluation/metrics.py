"""Offline prediction evaluation on the held-out recorded-action set.

For one model condition and one prediction set, every eligible origin
(registered from truth and metadata before any model output existed) is
rolled open loop along the logged actions at the declared horizons.
Ensemble members roll independently from the same truth origin; the
main prediction at each horizon is the arithmetic mean of the member
states, and members are retained only as diagnostics.

Failure accounting is two-part and never silently drops anything: a
non-finite prediction or a validity-bound crossing marks the origin
invalid at that and all later horizons (reported as the all-origin
invalid fraction), while RMSE is conditional on the finite origins and
always reported beside its numerator and denominator. A training-failed
condition still produces a structurally complete artifact with an
all-false validity mask and no fabricated numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from residual_worlds.config import ExperimentContract
from residual_worlds.data.dataset import load_dataset, load_segment_registry
from residual_worlds.evaluation.manifest import PredictionJob
from residual_worlds.models.base import Stepper
from residual_worlds.models.condition import load_condition_members
from residual_worlds.paths import prediction_job_dir
from residual_worlds.physics.kinematics import end_effector_position
from residual_worlds.provenance import is_complete, verify_artifact, write_artifact

# Validity bounds for open-loop rollouts: generous multiples of the
# physical ranges -- crossing them marks the prediction invalid while
# finite out-of-bound values still keep their numerical error.
_ANGLE_BOUND_MARGIN_RAD = 2.0
_VELOCITY_BOUND_FACTOR = 4.0


def _normalized_error(delta: np.ndarray) -> np.ndarray:
    dq = delta[..., :2] / 1.0
    dqd = delta[..., 2:] / 4.0
    return np.sqrt(0.5 * np.sum(dq**2, axis=-1) + 0.5 * np.sum(dqd**2, axis=-1))


def run_prediction_job(
    contract: ExperimentContract,
    job: PredictionJob,
    condition_directory: Path,
    prediction_set_directory: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    destination = prediction_job_dir(job.job_id)
    if is_complete(destination):
        verify_artifact(destination)
        return {"job_id": job.job_id, "artifact": str(destination), "reused": True}

    view = load_dataset(prediction_set_directory)
    registry = load_segment_registry(prediction_set_directory)
    horizons = sorted(int(h) for h in contract.data.rollout_horizons)
    payload = json.loads(
        (condition_directory / "condition.json").read_text(encoding="utf-8")
    )
    condition_failed = payload["status"] != "READY"

    stepper = Stepper.from_contract(contract)
    arm = contract.arm
    rows: dict[str, list[Any]] = {
        "horizon": [],
        "origin_row": [],
        "prediction_available": [],
        "valid": [],
        "normalized_state_error": [],
        "end_effector_error_m": [],
    }
    per_horizon_summary: dict[str, Any] = {}

    if not condition_failed:
        _method, members = load_condition_members(contract, condition_directory, device)
        for horizon in horizons:
            horizon_origins = registry.get(horizon, np.array([], dtype=np.int64))
            if horizon_origins.size == 0:
                per_horizon_summary[str(horizon)] = {
                    "eligible": 0, "finite": 0, "invalid_fraction": None, "rmse": None,
                }
                continue
            initial = torch.from_numpy(view.state[horizon_origins]).to(torch.float32)
            actions = torch.from_numpy(
                np.stack(
                    [view.action[horizon_origins + h] for h in range(horizon)], axis=1
                )
            ).to(torch.float32)
            truth_final = view.next_state[horizon_origins + horizon - 1]

            member_states = []
            with torch.no_grad():
                for member in members:
                    state = initial
                    for h in range(horizon):
                        state = stepper.step(member, state, actions[:, h])
                    member_states.append(state)
            mean_state = torch.stack(member_states, dim=0).mean(dim=0).numpy()

            finite = np.isfinite(mean_state).all(axis=-1)
            q_low = np.asarray(arm.q_min_rad) - _ANGLE_BOUND_MARGIN_RAD
            q_high = np.asarray(arm.q_max_rad) + _ANGLE_BOUND_MARGIN_RAD
            qd_bound = _VELOCITY_BOUND_FACTOR * np.asarray(arm.speed_limit_rad_s)
            in_bounds = (
                (mean_state[:, :2] >= q_low).all(axis=-1)
                & (mean_state[:, :2] <= q_high).all(axis=-1)
                & (np.abs(mean_state[:, 2:]) <= qd_bound).all(axis=-1)
            )
            valid = finite & in_bounds
            error = np.full(horizon_origins.shape[0], np.nan)
            ee_error = np.full(horizon_origins.shape[0], np.nan)
            if finite.any():
                error[finite] = _normalized_error(
                    mean_state[finite] - truth_final[finite]
                )
                predicted_ee = end_effector_position(
                    torch.from_numpy(mean_state[finite, :2]), arm
                ).numpy()
                truth_ee = end_effector_position(
                    torch.from_numpy(truth_final[finite, :2]), arm
                ).numpy()
                ee_error[finite] = np.linalg.norm(predicted_ee - truth_ee, axis=-1)

            for index, origin in enumerate(horizon_origins):
                rows["horizon"].append(horizon)
                rows["origin_row"].append(int(origin))
                rows["prediction_available"].append(True)
                rows["valid"].append(bool(valid[index]))
                rows["normalized_state_error"].append(
                    float(error[index]) if np.isfinite(error[index]) else None
                )
                rows["end_effector_error_m"].append(
                    float(ee_error[index]) if np.isfinite(ee_error[index]) else None
                )
            finite_errors = error[valid]
            per_horizon_summary[str(horizon)] = {
                "eligible": int(horizon_origins.shape[0]),
                "finite": int(valid.sum()),
                "invalid_fraction": float(1.0 - valid.mean()),
                "rmse": float(np.sqrt(np.mean(finite_errors**2)))
                if finite_errors.size
                else None,
            }
    else:
        for horizon in horizons:
            horizon_origins = registry.get(horizon, np.array([], dtype=np.int64))
            for origin in horizon_origins:
                rows["horizon"].append(horizon)
                rows["origin_row"].append(int(origin))
                rows["prediction_available"].append(False)
                rows["valid"].append(False)
                rows["normalized_state_error"].append(None)
                rows["end_effector_error_m"].append(None)
            per_horizon_summary[str(horizon)] = {
                "eligible": int(horizon_origins.shape[0]),
                "finite": 0,
                "invalid_fraction": 1.0 if horizon_origins.size else None,
                "rmse": None,
                "reason": "condition_training_failed",
            }

    summary = {
        "job_id": job.job_id,
        "method": job.method,
        "world_id": job.world_id,
        "budget": job.budget,
        "replicate": job.replicate,
        "condition_status": payload["status"],
        "per_horizon": per_horizon_summary,
    }

    def populate(directory: Path) -> None:
        pq.write_table(pa.table(rows), directory / "segment_metrics.parquet")
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    write_artifact(
        destination,
        "prediction_job",
        {"job_id": job.job_id, "method": job.method},
        {"condition": condition_directory.name, "prediction_set": prediction_set_directory.name},
        populate,
    )
    return {"job_id": job.job_id, "artifact": str(destination), "reused": False,
            "per_horizon": per_horizon_summary}
