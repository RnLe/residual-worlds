"""Target-world dataset generation: collection units, resets, accounting.

One **collection unit** is a fixed number of valid transitions assigned
to one excitation component and one split. Ordinarily it is one
contiguous trajectory from a saved safe reset; if a hard terminal event
occurs early, the terminal transition is kept (when numerically valid),
the segment is closed, and the unit continues from its deterministic
continuation reset with a fresh trajectory id. Nothing is padded and no
failure is deleted -- the budget counts exactly the valid transitions,
and the gross interaction (aborts, resets, discarded non-finite
endpoints) is logged separately so "data efficient" never hides
collection cost.

Free-excitation units (random, multisine) run in an empty workspace:
only joint-angle, joint-speed, and finiteness termination applies, and
resets follow a balanced schedule over the declared angle-band by
velocity-regime strata. Nominal-MPC units run on their assigned
training-task scenario (with its obstacle) under the nominal-model
controller plus a held perturbation, and reset to the scenario's own
initial state.

The learned methods later see only (x, u, x') rows -- the hidden world
parameters used here stay inside this generator and the artifact's
provenance section, which models never read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from residual_worlds.config import ExperimentContract
from residual_worlds.data.exploration import (
    band_limited_random_commands,
    multisine_commands,
    perturbation_commands,
)
from residual_worlds.data.split import (
    COMPONENT_ORDER,
    balanced_order,
    budget_membership,
    build_unit_plan,
)
from residual_worlds.identity import content_id
from residual_worlds.models.base import Stepper
from residual_worlds.models.nominal import NominalModel
from residual_worlds.paths import dataset_dir, prediction_set_dir
from residual_worlds.physics.integrators import AccelerationFn
from residual_worlds.physics.target import resolve_world, target_acceleration
from residual_worlds.planning.costs import ScenarioTensors
from residual_worlds.planning.mpc import ControllerSettings, MPCController, make_noise_fn
from residual_worlds.provenance import is_complete, verify_artifact, write_artifact
from residual_worlds.seeds import numpy_generator
from residual_worlds.task.geometry import swept_transition_check
from residual_worlds.task.reaching import TaskRules, TrueArmEnv
from residual_worlds.task.scenarios import load_bank
from residual_worlds.types import Scenario, TaskState

_FAR_OBSTACLE = torch.tensor([1000.0, 1000.0], dtype=torch.float64)


@dataclass
class _Accounting:
    attempted_transitions: int = 0
    retained_transitions: int = 0
    invalid_endpoints: int = 0
    aborted_segments: int = 0
    resets: int = 0
    terminal_reasons: dict[str, int] = field(default_factory=dict)

    def terminal(self, reason: str) -> None:
        self.terminal_reasons[reason] = self.terminal_reasons.get(reason, 0) + 1

    def to_payload(self, control_dt_s: float) -> dict[str, Any]:
        return {
            "attempted_transitions": self.attempted_transitions,
            "retained_transitions": self.retained_transitions,
            "invalid_endpoints": self.invalid_endpoints,
            "aborted_segments": self.aborted_segments,
            "resets": self.resets,
            "terminal_reasons": dict(sorted(self.terminal_reasons.items())),
            "gross_simulated_duration_s": self.attempted_transitions * control_dt_s,
            "retained_simulated_duration_s": self.retained_transitions * control_dt_s,
        }


@dataclass
class _Rows:
    state: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    next_state: list[np.ndarray] = field(default_factory=list)
    unit_id: list[int] = field(default_factory=list)
    trajectory_id: list[int] = field(default_factory=list)
    step_index: list[int] = field(default_factory=list)
    component_code: list[int] = field(default_factory=list)


@dataclass
class _TrajectoryRecord:
    trajectory_id: int
    unit_id: int
    start_row: int
    transition_count: int
    component_code: int
    terminal_reason: str


def _reset_schedules(
    contract: ExperimentContract, unit_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Balanced (angle-combo, velocity-regime) schedule per free unit."""
    strata = contract.data.reset_strata
    combo_weights = tuple(
        w1 * w2 for w1 in strata.shoulder_weights for w2 in strata.elbow_weights
    )
    return (
        balanced_order(unit_count, combo_weights),
        balanced_order(unit_count, tuple(strata.velocity_weights)),
    )


def _draw_reset(
    contract: ExperimentContract,
    rng: np.random.Generator,
    combo: int,
    velocity_regime: int,
    sign_flip: bool,
) -> np.ndarray:
    strata = contract.data.reset_strata
    shoulder_band = strata.shoulder_bands_rad[combo // len(strata.elbow_bands_rad)]
    elbow_band = strata.elbow_bands_rad[combo % len(strata.elbow_bands_rad)]
    arm = contract.arm
    for _ in range(1000):
        q1 = rng.uniform(*shoulder_band)
        q2 = rng.uniform(*elbow_band)
        if velocity_regime == 0:
            qd = rng.uniform(
                -strata.velocity_low_abs_max_rad_s, strata.velocity_low_abs_max_rad_s, size=2
            )
        else:
            low, high = strata.velocity_moderate_abs_range_rad_s
            magnitude = rng.uniform(low, high, size=2)
            signs = np.array([1.0, -1.0] if not sign_flip else [-1.0, 1.0])
            qd = magnitude * signs
        state = np.array([q1, q2, qd[0], qd[1]], dtype=np.float64)
        inside = all(
            arm.q_min_rad[j] < state[j] < arm.q_max_rad[j] for j in range(2)
        ) and all(abs(state[2 + j]) < arm.speed_limit_rad_s[j] for j in range(2))
        if inside and np.isfinite(state).all():
            return state
    raise RuntimeError("could not draw a valid reset state inside the declared band")


def _free_transition_terminal(
    endpoints: torch.Tensor, contract: ExperimentContract, h: float
) -> str | None:
    """Terminal reason for one free-excitation transition, if any."""
    if not bool(torch.isfinite(endpoints).all()):
        return "NONFINITE"
    check = swept_transition_check(
        endpoints[:-1, :2],
        endpoints[:-1, 2:],
        endpoints[1:, :2],
        endpoints[1:, 2:],
        h,
        _FAR_OBSTACLE,
        0.0,
        contract.task.arm_safety_radius_m,
        contract.task.swept_collision_inflation_m,
        contract.arm,
    )
    if bool(check.limit_violation.any()):
        return "HARD_LIMIT_OR_SPEED"
    return None


def _collect_free_unit(
    contract: ExperimentContract,
    accel: AccelerationFn,
    world_id: str,
    purpose: str,
    replicate: int,
    unit_index: int,
    component_code: int,
    combo: int,
    velocity_regime: int,
    rows: _Rows,
    trajectories: list[_TrajectoryRecord],
    accounting: _Accounting,
    next_trajectory_id: int,
) -> int:
    unit_size = contract.data.collection_unit_valid_transitions
    stepper = Stepper.from_contract(contract)
    rng = numpy_generator(
        contract.numerics.root_seed, purpose, world_id, replicate, unit_index, "commands"
    )
    reset_rng = numpy_generator(
        contract.numerics.root_seed, purpose, world_id, replicate, unit_index, "resets"
    )
    config = contract.data.exploration
    horizon = 8 * unit_size  # generous command reservoir; never regenerated
    if COMPONENT_ORDER[component_code] == "band_limited_random":
        commands = band_limited_random_commands(rng, horizon, config)
    else:
        commands = multisine_commands(rng, horizon, config, contract.numerics.control_dt_s)

    valid = 0
    command_cursor = 0
    sign_flip = unit_index % 2 == 1
    state = torch.from_numpy(
        _draw_reset(contract, reset_rng, combo, velocity_regime, sign_flip)
    )
    accounting.resets += 1
    segment_start_row = len(rows.state)
    segment_steps = 0
    trajectory_id = next_trajectory_id
    next_trajectory_id += 1
    h = contract.numerics.control_dt_s / contract.numerics.substeps_per_control_step

    while valid < unit_size:
        if command_cursor >= commands.shape[0]:
            raise RuntimeError("command reservoir exhausted; unit cannot complete")
        action = torch.from_numpy(commands[command_cursor])
        command_cursor += 1
        endpoints = stepper.substep_endpoints(accel, state, action)
        accounting.attempted_transitions += 1
        terminal = _free_transition_terminal(endpoints, contract, h)
        next_state = endpoints[-1]
        finite = bool(torch.isfinite(next_state).all()) and bool(
            torch.isfinite(state).all()
        )
        record = terminal != "NONFINITE" and finite
        if record:
            rows.state.append(state.numpy().copy())
            rows.action.append(action.numpy().copy())
            rows.next_state.append(next_state.numpy().copy())
            rows.unit_id.append(unit_index)
            rows.trajectory_id.append(trajectory_id)
            rows.step_index.append(segment_steps)
            rows.component_code.append(component_code)
            valid += 1
            segment_steps += 1
            accounting.retained_transitions += 1
        else:
            accounting.invalid_endpoints += 1
        if terminal is not None:
            accounting.terminal(terminal)
            accounting.aborted_segments += 1
            trajectories.append(
                _TrajectoryRecord(
                    trajectory_id, unit_index, segment_start_row, segment_steps,
                    component_code, terminal,
                )
            )
            state = torch.from_numpy(
                _draw_reset(contract, reset_rng, combo, velocity_regime, sign_flip)
            )
            accounting.resets += 1
            segment_start_row = len(rows.state)
            segment_steps = 0
            trajectory_id = next_trajectory_id
            next_trajectory_id += 1
        else:
            state = next_state
    if segment_steps > 0:
        trajectories.append(
            _TrajectoryRecord(
                trajectory_id, unit_index, segment_start_row, segment_steps,
                component_code, "UNIT_COMPLETE",
            )
        )
    return next_trajectory_id


def _collect_mpc_unit(
    contract: ExperimentContract,
    target_accel: AccelerationFn,
    scenario: Scenario,
    world_id: str,
    purpose: str,
    replicate: int,
    unit_index: int,
    rows: _Rows,
    trajectories: list[_TrajectoryRecord],
    accounting: _Accounting,
    next_trajectory_id: int,
) -> int:
    """Nominal-MPC rollout on a training-task scenario, plus perturbation."""
    unit_size = contract.data.collection_unit_valid_transitions
    component_code = COMPONENT_ORDER.index("nominal_mpc")
    config = contract.data.exploration
    rng = numpy_generator(
        contract.numerics.root_seed, purpose, world_id, replicate, unit_index, "perturbation"
    )
    perturbations = perturbation_commands(rng, 8 * unit_size, config)
    envelope = np.asarray(config.torque_envelope_nm)

    rules = TaskRules.from_config(contract.task)
    nominal_model = NominalModel(contract.arm)
    settings = ControllerSettings.from_planning(contract.planning)
    shape = (
        contract.planning.iterations,
        contract.planning.candidates,
        contract.planning.action_knots,
        2,
    )
    controller = MPCController(
        [nominal_model.acceleration],
        ScenarioTensors.from_scenario(scenario, torch.float32),
        contract.task.cost,
        rules,
        contract.arm,
        Stepper.from_contract(contract),
        settings,
        make_noise_fn(
            contract.numerics.root_seed,
            ("collection_cem", purpose, world_id, replicate, unit_index),
            shape,
        ),
    )
    env = TrueArmEnv(
        target_accel,
        scenario,
        rules,
        contract.arm,
        contract.numerics.control_dt_s,
        contract.numerics.substeps_per_control_step,
    )

    valid = 0
    cursor = 0
    env.reset()
    controller.reset()
    previous = (0.0, 0.0)
    segment_start_row = len(rows.state)
    segment_steps = 0
    trajectory_id = next_trajectory_id
    next_trajectory_id += 1
    accounting.resets += 1
    mpc_step = 0

    while valid < unit_size:
        if cursor >= perturbations.shape[0]:
            raise RuntimeError("perturbation reservoir exhausted; unit cannot complete")
        state = torch.from_numpy(env.state)
        task = TaskState(env.target_index, env.dwell_count, previous)
        plan = controller.plan(state, task, mpc_step)
        mpc_step += 1
        command = plan.actions[0].to(torch.float64).numpy() + perturbations[cursor]
        cursor += 1
        command = np.clip(command, -envelope, envelope)
        before = env.state
        _obs, _reward, terminated, truncated, info = env.step(command)
        accounting.attempted_transitions += 1
        after = env.state
        finite = bool(np.isfinite(before).all() and np.isfinite(after).all())
        reason = info.get("reason")
        if finite and reason != "NONFINITE_OR_MODEL_ERROR":
            rows.state.append(before.copy())
            rows.action.append(command.copy())
            rows.next_state.append(after.copy())
            rows.unit_id.append(unit_index)
            rows.trajectory_id.append(trajectory_id)
            rows.step_index.append(segment_steps)
            rows.component_code.append(component_code)
            valid += 1
            segment_steps += 1
            accounting.retained_transitions += 1
        else:
            accounting.invalid_endpoints += 1
        previous = (float(command[0]), float(command[1]))
        if terminated or truncated:
            accounting.terminal(str(reason))
            accounting.aborted_segments += 1
            trajectories.append(
                _TrajectoryRecord(
                    trajectory_id, unit_index, segment_start_row, segment_steps,
                    component_code, str(reason),
                )
            )
            env.reset()
            controller.reset()
            previous = (0.0, 0.0)
            accounting.resets += 1
            segment_start_row = len(rows.state)
            segment_steps = 0
            trajectory_id = next_trajectory_id
            next_trajectory_id += 1
    if segment_steps > 0:
        trajectories.append(
            _TrajectoryRecord(
                trajectory_id, unit_index, segment_start_row, segment_steps,
                component_code, "UNIT_COMPLETE",
            )
        )
    return next_trajectory_id


def _dataset_spec(
    contract: ExperimentContract, world_id: str, replicate: int, kind: str, units: int
) -> dict[str, Any]:
    exploration = contract.data.exploration
    return {
        "schema": 1,
        "kind": kind,
        "world_id": world_id,
        "replicate": replicate,
        "root_seed": contract.numerics.root_seed,
        "control_dt_s": contract.numerics.control_dt_s,
        "substeps": contract.numerics.substeps_per_control_step,
        "unit_size": contract.data.collection_unit_valid_transitions,
        "units": units,
        "budgets": list(contract.data.adaptation_budgets_total),
        "train_fraction": contract.data.train_fraction,
        "exploration": {
            "fractions": [
                exploration.band_limited_random_fraction,
                exploration.multisine_fraction,
                exploration.nominal_mpc_perturbed_fraction,
            ],
            "envelope": list(exploration.torque_envelope_nm),
            "cutoff_hz": exploration.random_cutoff_hz,
            "frequencies_hz": list(exploration.multisine_frequencies_hz),
            "perturbation": [
                exploration.mpc_perturbation_low_nm,
                exploration.mpc_perturbation_high_nm,
                exploration.mpc_perturbation_hold_steps,
            ],
        },
    }


def _write_parquet(path: Path, columns: dict[str, list[Any]]) -> None:
    pq.write_table(pa.table(columns), path)


def _generate(
    contract: ExperimentContract,
    world_id: str,
    replicate: int,
    scenario_dir: Path,
    kind: str,
) -> dict[str, Any]:
    """Shared machinery for adaptation datasets and prediction sets."""
    data = contract.data
    unit_size = data.collection_unit_valid_transitions
    if kind == "adaptation":
        max_budget = max(data.adaptation_budgets_total)
        plan = build_unit_plan(
            unit_size,
            max_budget,
            data.train_fraction,
            (
                data.exploration.band_limited_random_fraction,
                data.exploration.multisine_fraction,
                data.exploration.nominal_mpc_perturbed_fraction,
            ),
        )
        components = plan.train_components + plan.validation_components
        purpose = "data"
        destination = dataset_dir(
            content_id(
                "dataset", _dataset_spec(contract, world_id, replicate, kind, len(components))
            )
        )
    else:
        units = data.prediction_test_transitions // unit_size
        components = balanced_order(
            units,
            (
                data.exploration.band_limited_random_fraction,
                data.exploration.multisine_fraction,
                data.exploration.nominal_mpc_perturbed_fraction,
            ),
        )
        plan = None
        purpose = "prediction_set"
        destination = prediction_set_dir(
            content_id(
                "prediction_set",
                _dataset_spec(contract, world_id, replicate, kind, len(components)),
            )
        )

    if is_complete(destination):
        verify_artifact(destination)
        return {"artifact": str(destination), "reused": True}

    world = resolve_world(contract, world_id)
    arm = contract.arm

    def target_accel(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return target_acceleration(state, action, world, arm)

    task_scenarios = load_bank(scenario_dir / "training_task.json")
    combos, regimes = _reset_schedules(contract, len(components))

    rows = _Rows()
    trajectories: list[_TrajectoryRecord] = []
    accounting = _Accounting()
    next_trajectory = 0
    mpc_code = COMPONENT_ORDER.index("nominal_mpc")
    for unit_index, component_code in enumerate(components):
        if component_code == mpc_code:
            scenario = task_scenarios[unit_index % len(task_scenarios)]
            next_trajectory = _collect_mpc_unit(
                contract, target_accel, scenario, world_id, purpose, replicate,
                unit_index, rows, trajectories, accounting, next_trajectory,
            )
        else:
            next_trajectory = _collect_free_unit(
                contract, target_accel, world_id, purpose, replicate, unit_index,
                component_code, combos[unit_index], regimes[unit_index],
                rows, trajectories, accounting, next_trajectory,
            )

    state = np.stack(rows.state)
    action = np.stack(rows.action)
    next_state = np.stack(rows.next_state)

    def populate(directory: Path) -> None:
        np.savez(
            directory / "transitions.npz",
            state=state,
            action=action,
            next_state=next_state,
            collection_unit_id=np.asarray(rows.unit_id, dtype=np.int64),
            trajectory_id=np.asarray(rows.trajectory_id, dtype=np.int64),
            step_index=np.asarray(rows.step_index, dtype=np.int32),
            component_code=np.asarray(rows.component_code, dtype=np.int16),
        )
        _write_parquet(
            directory / "collection_units.parquet",
            {
                "unit_id": list(range(len(components))),
                "component_code": [int(c) for c in components],
                "split": (
                    ["train"] * plan.train_units + ["validation"] * plan.validation_units
                    if plan is not None
                    else ["prediction_test"] * len(components)
                ),
                "reset_combo": [int(c) for c in combos],
                "velocity_regime": [int(r) for r in regimes],
                "transitions": [unit_size] * len(components),
            },
        )
        _write_parquet(
            directory / "trajectories.parquet",
            {
                "trajectory_id": [t.trajectory_id for t in trajectories],
                "unit_id": [t.unit_id for t in trajectories],
                "start_row": [t.start_row for t in trajectories],
                "transition_count": [t.transition_count for t in trajectories],
                "component_code": [t.component_code for t in trajectories],
                "terminal_reason": [t.terminal_reason for t in trajectories],
            },
        )
        if plan is not None:
            import json

            membership = budget_membership(
                data.adaptation_budgets_total, unit_size, data.train_fraction, plan
            )
            (directory / "split.json").write_text(
                json.dumps(
                    {
                        "train_units": list(range(plan.train_units)),
                        "validation_units": [
                            plan.train_units + i for i in range(plan.validation_units)
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_parquet(
                directory / "budget_membership.parquet",
                {
                    "budget": [b for b in membership for _ in range(2)],
                    "split": ["train", "validation"] * len(membership),
                    "unit_ids": [
                        list(units)
                        for b in membership
                        for units in membership[b]
                    ],
                },
            )
        else:
            _write_segment_registry(directory, contract, rows)

    write_artifact(
        destination,
        f"{kind}_dataset",
        {
            "world_id": world_id,
            "replicate": replicate,
            "kind": kind,
        },
        {"contract": contract.source_path},
        populate,
        extra_manifest={
            "accounting": accounting.to_payload(contract.numerics.control_dt_s),
            # Hidden world parameters live only here, in provenance,
            # for the evaluator; model loaders never read the manifest.
            "world_provenance": {
                "payload_kg": world.payload_kg,
                "has_friction": world.friction is not None,
                "has_actuator": world.actuator is not None,
                "elastic_coupling_nm": world.elastic_coupling_nm,
            },
        },
    )
    return {
        "artifact": str(destination),
        "reused": False,
        "transitions": int(state.shape[0]),
        "accounting": accounting.to_payload(contract.numerics.control_dt_s),
    }


def _write_segment_registry(
    directory: Path, contract: ExperimentContract, rows: _Rows
) -> None:
    """Eligible open-loop windows per horizon (truth/metadata only)."""
    trajectory = np.asarray(rows.trajectory_id)
    step = np.asarray(rows.step_index)
    horizons: list[int] = []
    origins: list[int] = []
    count = trajectory.shape[0]
    for horizon in contract.data.rollout_horizons:
        for origin in range(count):
            end = origin + horizon - 1
            if end >= count:
                continue
            window_trajectory = trajectory[origin : end + 1]
            window_steps = step[origin : end + 1]
            if (window_trajectory == window_trajectory[0]).all() and (
                np.diff(window_steps) == 1
            ).all():
                horizons.append(horizon)
                origins.append(origin)
    _write_parquet(
        directory / "segment_registry.parquet",
        {"horizon": horizons, "origin_row": origins},
    )


def generate_world_dataset(
    contract: ExperimentContract, world_id: str, replicate: int, scenario_dir: Path
) -> dict[str, Any]:
    return _generate(contract, world_id, replicate, scenario_dir, "adaptation")


def generate_prediction_set(
    contract: ExperimentContract, world_id: str, replicate: int, scenario_dir: Path
) -> dict[str, Any]:
    return _generate(contract, world_id, replicate, scenario_dir, "prediction")
