"""Golden numbers pinning the site's TypeScript physics to this package.

The site draws the preview live from a hand-written mirror of the
nominal and target dynamics. This fixture holds parameters and sample
evaluations from the Python side so a test on each side can fail the
moment the two drift apart.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from residual_worlds.config import ExperimentContract
from residual_worlds.media.loop import SCHEDULE, Schedule, reference, simulate_loop, tracking_torque
from residual_worlds.physics import nominal
from residual_worlds.physics.integrators import rk4_transition
from residual_worlds.physics.kinematics import elbow_position, end_effector_position
from residual_worlds.physics.target import resolve_world, target_acceleration

# Sample points chosen to exercise every branch: rest, slow motion inside
# the friction smoothing band, fast motion, and commands inside and
# outside the actuator dead zone.
SAMPLE_STATES = (
    (1.6, -1.0, 0.0, 0.0),
    (1.6, -1.0, 0.02, -0.03),
    (1.2, -0.4, 0.8, -1.5),
    (2.1, -1.8, -2.5, 3.0),
    (0.9, 0.6, 3.0, -0.5),
    (1.75, -1.15, 0.5, 0.5),
)
SAMPLE_ACTIONS = (
    (0.0, 0.0),
    (0.05, -0.05),
    (1.5, -0.4),
    (-3.0, 2.0),
    (4.0, -4.0),
    (2.2, 0.1),
)


def _rows(tensor: torch.Tensor) -> list[Any]:
    rows: list[Any] = tensor.tolist()
    return rows


def build_arm_golden(
    contract: ExperimentContract,
    world_id: str = "composite_standard",
    schedule: Schedule = SCHEDULE,
) -> dict[str, Any]:
    arm = contract.arm
    world = resolve_world(contract, world_id)
    dt = contract.numerics.control_dt_s
    substeps = contract.numerics.substeps_per_control_step

    def true_acc(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return target_acceleration(s, a, world, arm)

    def nominal_acc(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return nominal.state_acceleration(s, a, arm)

    states = torch.tensor(SAMPLE_STATES, dtype=torch.float64)
    actions = torch.tensor(SAMPLE_ACTIONS, dtype=torch.float64)
    q = states[:, :2]

    samples = []
    for i in range(states.shape[0]):
        samples.append(
            {
                "state": _rows(states[i]),
                "action": _rows(actions[i]),
                "elbow": _rows(elbow_position(q[i], arm)),
                "hand": _rows(end_effector_position(q[i], arm)),
                "nominal_acc": _rows(nominal_acc(states[i], actions[i])),
                "target_acc": _rows(true_acc(states[i], actions[i])),
                "tracking_torque": _rows(
                    tracking_torque(states[i], 0.37 * i, arm, world, schedule)
                ),
            }
        )

    rollouts = []
    for i in (0, 2, 3):
        nominal_path = [states[i]]
        target_path = [states[i]]
        for _ in range(schedule.ghost_steps):
            nominal_path.append(
                rk4_transition(nominal_acc, nominal_path[-1], actions[i], dt, substeps)
            )
            target_path.append(rk4_transition(true_acc, target_path[-1], actions[i], dt, substeps))
        rollouts.append(
            {
                "state": _rows(states[i]),
                "action": _rows(actions[i]),
                "nominal": [_rows(s) for s in nominal_path],
                "target": [_rows(s) for s in target_path],
            }
        )

    loop = simulate_loop(contract, world_id, schedule)
    q_ref, qd_ref, qdd_ref = reference(1.234, schedule)

    return {
        "schema": 1,
        "world_id": world_id,
        "dt_s": dt,
        "substeps": substeps,
        "arm": asdict(arm),
        "world": {
            "payload_kg": world.payload_kg,
            "friction": None if world.friction is None else asdict(world.friction),
            "actuator": None if world.actuator is None else asdict(world.actuator),
            "elastic_coupling_nm": world.elastic_coupling_nm,
        },
        "schedule": asdict(schedule),
        "reference_at_1p234_s": {"q": q_ref, "qd": qd_ref, "qdd": qdd_ref},
        "samples": samples,
        "rollouts": rollouts,
        "loop": {
            "frames": loop.frames,
            "states_first": loop.states[0].tolist(),
            "states_last": loop.states[-1].tolist(),
            "actions_first": loop.actions[0].tolist(),
            "ghost_first_end": loop.ghosts[0, -1].tolist(),
            "residual_abs_max": [float(v) for v in abs(loop.residual).max(axis=0)],
        },
    }


def write_arm_golden(
    contract: ExperimentContract, destination: Path, world_id: str = "composite_standard"
) -> Path:
    payload = build_arm_golden(contract, world_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination
