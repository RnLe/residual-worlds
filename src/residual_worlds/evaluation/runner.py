"""Row-driven closed-loop evaluation in the true target world.

For one manifest row the runner constructs the hidden target world
(this is the one harness that may), loads the assigned model condition,
runs the frozen controller with the row's method-free CEM noise stream,
and writes one immutable evaluation artifact: exact states and actions,
per-call planner records, realized stage costs recomputed on truth,
events, and a summary with the exact failure precedence.

Scientific failures (numerical divergence, collisions, limits,
timeouts, a training-failed condition) are complete outcomes with their
own codes -- never reruns. Support distance against the method's active
training prefix is computed for the three adapted methods only; nominal
and the exact-dynamics reference have no target-data prefix and record
the metric as not applicable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from residual_worlds.config import ExperimentContract
from residual_worlds.data.dataset import load_dataset
from residual_worlds.evaluation.manifest import ControlJob
from residual_worlds.models.base import Stepper
from residual_worlds.models.condition import load_condition_members
from residual_worlds.models.normalization import PhysicalScales, features
from residual_worlds.paths import evaluation_dir
from residual_worlds.physics.target import resolve_world, target_acceleration
from residual_worlds.planning.costs import ScenarioTensors, transition_stage_cost
from residual_worlds.planning.mpc import ControllerSettings, MPCController, make_noise_fn
from residual_worlds.provenance import is_complete, verify_artifact, write_artifact
from residual_worlds.task.reaching import TaskRules, TrueArmEnv
from residual_worlds.types import Scenario, TaskState

_REASON_TO_CODE = {
    "SUCCESS": "SUCCESS",
    "NONFINITE_OR_MODEL_ERROR": "NONFINITE_OR_MODEL_ERROR",
    "HARD_LIMIT_OR_SPEED": "HARD_LIMIT_OR_SPEED",
    "OBSTACLE_COLLISION": "OBSTACLE_COLLISION",
    "TIMEOUT_ZERO_TARGETS": "TIMEOUT_ZERO_TARGETS",
    "TIMEOUT_PARTIAL_TARGETS": "TIMEOUT_PARTIAL_TARGETS",
}


def _noise_namespace(contract: ExperimentContract, job: ControlJob) -> tuple[str | int, ...]:
    protocol_tag = contract.protocol.version
    if job.noise_family == "composite_family":
        # Method, budget, and evaluation world deliberately omitted: the
        # budget and transfer contrasts share primitive randomness.
        return ("cem_panel", "composite_family", protocol_tag, job.replicate, job.scenario_id)
    return (
        "cem_call",
        "paired_methods",
        protocol_tag,
        job.world_id,
        job.budget,
        job.replicate,
        job.scenario_id,
    )


def _support_distances(
    contract: ExperimentContract,
    states: np.ndarray,
    actions: np.ndarray,
    dataset_directory: Path,
    budget: int,
) -> np.ndarray:
    """Exact 1-NN distance in normalized feature space to the train prefix."""
    view = load_dataset(dataset_directory)
    train_units, _ = view.units_for_budget(
        budget,
        contract.data.collection_unit_valid_transitions,
        contract.data.train_fraction,
    )
    rows = view.rows_for_units(train_units)
    scales = PhysicalScales.from_contract(contract)
    train_features = features(
        torch.from_numpy(view.state[rows]), torch.from_numpy(view.action[rows]), scales
    )
    executed_features = features(
        torch.from_numpy(states), torch.from_numpy(actions), scales
    )
    distances = torch.cdist(executed_features, train_features).min(dim=1).values
    return distances.numpy()


def run_control_job(
    contract: ExperimentContract,
    job: ControlJob,
    scenario: Scenario,
    condition_directory: Path | None,
    dataset_directory: Path | None,
    device: str = "cpu",
) -> dict[str, Any]:
    destination = evaluation_dir(job.job_id)
    if is_complete(destination):
        verify_artifact(destination)
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        return {"job_id": job.job_id, "artifact": str(destination), "reused": True,
                "success": summary["success"], "termination": summary["termination_code"]}

    world = resolve_world(contract, job.world_id)
    arm = contract.arm

    def target_accel(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return target_acceleration(state, action, world, arm)

    # Resolve the condition into planner members. A TRAINING_FAILED
    # condition materializes a structurally complete failed row.
    members: list[Any]
    if job.method == "oracle":
        members = [target_accel]
    elif job.method == "nominal":
        from residual_worlds.models.nominal import NominalModel

        members = [NominalModel(arm).acceleration]
    else:
        assert condition_directory is not None
        payload = json.loads(
            (condition_directory / "condition.json").read_text(encoding="utf-8")
        )
        if payload["status"] != "READY":
            return _write_training_failed_row(contract, job, scenario, destination, payload)
        _method, members = load_condition_members(contract, condition_directory, device)

    rules = TaskRules.from_config(contract.task)
    settings = ControllerSettings.from_planning(contract.planning)
    shape = (
        contract.planning.iterations,
        contract.planning.candidates,
        contract.planning.action_knots,
        2,
    )
    controller = MPCController(
        members,
        ScenarioTensors.from_scenario(scenario, torch.float32),
        contract.task.cost,
        rules,
        arm,
        Stepper.from_contract(contract),
        settings,
        make_noise_fn(contract.numerics.root_seed, _noise_namespace(contract, job), shape),
    )
    env = TrueArmEnv(
        target_accel,
        scenario,
        rules,
        arm,
        contract.numerics.control_dt_s,
        contract.numerics.substeps_per_control_step,
    )
    env.reset()
    controller.reset()

    scenario_tensors = ScenarioTensors.from_scenario(scenario, torch.float64)
    states = [env.state]
    actions: list[np.ndarray] = []
    realized_costs: list[float] = []
    target_indices = [env.target_index]
    dwell_counts = [env.dwell_count]
    mpc_call_of_step: list[int] = []
    clearances: list[float] = []
    events: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    previous_action = (0.0, 0.0)
    termination_reason: str | None = None
    call_index = -1

    while termination_reason is None:
        state = torch.from_numpy(env.state)
        task = TaskState(env.target_index, env.dwell_count, previous_action)
        call_index += 1
        started = time.perf_counter_ns()
        plan = controller.plan(state, task, env.executed_steps)
        planning_ns = time.perf_counter_ns() - started
        planner_rows.append(
            {
                "call_index": call_index,
                "physical_step": env.executed_steps,
                "noise_sha256": plan.diagnostics["noise_sha256"],
                "predicted_cost": plan.predicted_cost,
                "final_mean_invalid": bool(plan.diagnostics["final_mean_invalid"]),
                "predicted_min_clearance_m": plan.diagnostics["predicted_min_clearance"],
                "planning_time_ms": planning_ns / 1e6,
            }
        )
        if plan.diagnostics["plan_nonfinite"]:
            termination_reason = "NONFINITE_OR_MODEL_ERROR"
            events.append({"type": "nonfinite_plan", "step": env.executed_steps})
            break

        # Execute the planned prefix, checking terminal events after
        # every physical transition (profile C executes two actions).
        for action_tensor in plan.actions:
            action = action_tensor.detach().to(torch.float64).numpy()
            index_before = env.target_index
            _obs, _reward, terminated, truncated, info = env.step(action)
            actions.append(action.copy())
            states.append(env.state)
            target_indices.append(env.target_index)
            dwell_counts.append(env.dwell_count)
            mpc_call_of_step.append(call_index)
            clearance = float(info.get("min_clearance_m", float("nan")))
            clearances.append(clearance)
            events.extend(info.get("events", []))

            margins = info.get("min_joint_margin_rad", (float("nan"), float("nan")))
            realized = transition_stage_cost(
                torch.from_numpy(env.state),
                torch.from_numpy(action),
                torch.tensor(previous_action, dtype=torch.float64),
                torch.tensor(index_before),
                torch.tensor(clearance, dtype=torch.float64),
                torch.tensor(margins, dtype=torch.float64),
                scenario_tensors.targets,
                contract.task.cost,
                rules,
                arm,
            )
            realized_costs.append(float(realized))
            previous_action = (float(action[0]), float(action[1]))
            if terminated or truncated:
                termination_reason = str(info["reason"])
                break

    assert termination_reason is not None
    code = _REASON_TO_CODE[termination_reason]
    success = code == "SUCCESS"
    executed = len(actions)
    dt = contract.numerics.control_dt_s
    timeout_s = scenario.timeout_steps * dt

    states_array = np.stack(states)
    actions_array = (
        np.stack(actions) if actions else np.zeros((0, 2), dtype=np.float64)
    )
    support: np.ndarray | None = None
    if job.method in ("fitted_physics", "blackbox", "residual") and executed > 0:
        assert dataset_directory is not None
        support = _support_distances(
            contract, states_array[:-1], actions_array, dataset_directory, job.budget
        )

    planning_times = [row["planning_time_ms"] for row in planner_rows]
    summary = {
        "job_id": job.job_id,
        "method": job.method,
        "world_id": job.world_id,
        "budget": job.budget,
        "replicate": job.replicate,
        "scenario_id": job.scenario_id,
        "panel_memberships": list(job.panel_memberships),
        "success": success,
        "termination_code": code,
        "executed_steps": executed,
        "targets_completed": int(target_indices[-1]),
        "completion_time_restricted_s": executed * dt if success else timeout_s,
        "realized_cost_total": float(np.sum(realized_costs)) if realized_costs else 0.0,
        "predicted_cost_first_call": planner_rows[0]["predicted_cost"] if planner_rows else None,
        "torque_effort": float(np.sum(actions_array**2) * dt),
        "action_variation": float(
            np.sum(np.diff(actions_array, axis=0) ** 2) if executed > 1 else 0.0
        ),
        "min_clearance_m": float(np.nanmin(clearances)) if clearances else None,
        "support_distance_mean": float(support.mean()) if support is not None else None,
        "support_distance_max": float(support.max()) if support is not None else None,
        "support_not_applicable_reason": (
            None if support is not None else "no_target_data_prefix"
        ),
        "planning_time_ms_p50": float(np.percentile(planning_times, 50)),
        "planning_time_ms_p95": float(np.percentile(planning_times, 95)),
    }

    def populate(directory: Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "state": states_array,
            "action": actions_array,
            "realized_stage_cost": np.asarray(realized_costs, dtype=np.float64),
            "target_index": np.asarray(target_indices, dtype=np.int8),
            "dwell_count": np.asarray(dwell_counts, dtype=np.int16),
            "mpc_call_index": np.asarray(mpc_call_of_step, dtype=np.int32),
            "minimum_clearance_m": np.asarray(clearances, dtype=np.float64),
        }
        if support is not None:
            arrays["support_distance"] = support
        np.savez(directory / "trajectory.npz", **arrays)  # type: ignore[arg-type]
        pq.write_table(
            pa.table({key: [row[key] for row in planner_rows] for key in planner_rows[0]}),
            directory / "planner_calls.parquet",
        )
        (directory / "events.json").write_text(
            json.dumps(events, indent=2) + "\n", encoding="utf-8"
        )
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    write_artifact(
        destination,
        "evaluation_rollout",
        {"job_id": job.job_id, "method": job.method, "scenario_id": job.scenario_id},
        {
            "condition": condition_directory.name if condition_directory else None,
            "contract": contract.source_path,
        },
        populate,
    )
    return {
        "job_id": job.job_id,
        "artifact": str(destination),
        "reused": False,
        "success": success,
        "termination": code,
    }


def _write_training_failed_row(
    contract: ExperimentContract,
    job: ControlJob,
    scenario: Scenario,
    destination: Path,
    condition_payload: dict[str, Any],
) -> dict[str, Any]:
    """Structurally complete failed row: no fabricated trajectory."""
    dt = contract.numerics.control_dt_s
    summary = {
        "job_id": job.job_id,
        "method": job.method,
        "world_id": job.world_id,
        "budget": job.budget,
        "replicate": job.replicate,
        "scenario_id": job.scenario_id,
        "panel_memberships": list(job.panel_memberships),
        "success": False,
        "termination_code": "TRAINING_FAILED",
        "executed_steps": 0,
        "targets_completed": 0,
        "completion_time_restricted_s": scenario.timeout_steps * dt,
        "realized_cost_total": None,
        "predicted_cost_first_call": None,
        "torque_effort": None,
        "action_variation": None,
        "min_clearance_m": None,
        "support_distance_mean": None,
        "support_distance_max": None,
        "support_not_applicable_reason": "condition_training_failed",
        "planning_time_ms_p50": None,
        "planning_time_ms_p95": None,
    }

    def populate(directory: Path) -> None:
        np.savez(
            directory / "trajectory.npz",
            state=np.asarray([scenario.initial_state], dtype=np.float64),
            action=np.zeros((0, 2), dtype=np.float64),
            realized_stage_cost=np.zeros((0,), dtype=np.float64),
            target_index=np.zeros((1,), dtype=np.int8),
            dwell_count=np.zeros((1,), dtype=np.int16),
            mpc_call_index=np.zeros((0,), dtype=np.int32),
            minimum_clearance_m=np.zeros((0,), dtype=np.float64),
        )
        (directory / "events.json").write_text(
            json.dumps(
                [{"type": "training_failed",
                  "reason": condition_payload.get("failure_reason")}],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    write_artifact(
        destination,
        "evaluation_rollout",
        {"job_id": job.job_id, "method": job.method, "scenario_id": job.scenario_id},
        {"condition": condition_payload["condition_id"], "contract": contract.source_path},
        populate,
    )
    return {
        "job_id": job.job_id,
        "artifact": str(destination),
        "reused": False,
        "success": False,
        "termination": "TRAINING_FAILED",
    }
