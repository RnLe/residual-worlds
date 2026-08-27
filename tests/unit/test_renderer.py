"""Renderer determinism and world-to-pixel consistency."""

import hashlib

import torch

from residual_worlds.config import load_contract
from residual_worlds.paths import repository_root
from residual_worlds.physics.kinematics import end_effector_position
from residual_worlds.task.renderer import WorldToPixel, render_state
from residual_worlds.types import Scenario

CONTRACT = load_contract(repository_root() / "configs" / "experiment_contract.yaml")

SCENARIO = Scenario(
    scenario_id="render-test",
    bank="test",
    index=0,
    stratum_id=0,
    target_order=(0, 1, 2),
    obstacle_chord_index=0,
    initial_state=(0.9, -0.4, 0.0, 0.0),
    targets_xy_m=((0.4, 0.5), (-0.2, 0.6), (0.1, 0.3)),
    obstacle_xy_radius_m=(0.1, 0.45, 0.08),
    timeout_steps=160,
)


def _render_bytes() -> bytes:
    image = render_state(
        torch.tensor([0.9, -0.4], dtype=torch.float64),
        SCENARIO,
        CONTRACT.arm,
        width=640,
        height=360,
        target_index=1,
        method_label="nominal physics",
        arm_safety_radius_m=CONTRACT.task.arm_safety_radius_m,
    )
    return image.tobytes()


def test_rendering_is_deterministic() -> None:
    assert hashlib.sha256(_render_bytes()).hexdigest() == hashlib.sha256(
        _render_bytes()
    ).hexdigest()


def test_end_effector_lands_on_transform_position() -> None:
    q = torch.tensor([0.9, -0.4], dtype=torch.float64)
    ee = end_effector_position(q, CONTRACT.arm)
    transform = WorldToPixel(width=640, height=360)
    px, py = transform.to_pixel(float(ee[0]), float(ee[1]))
    image = render_state(
        q,
        SCENARIO,
        CONTRACT.arm,
        width=640,
        height=360,
        target_index=0,
        method_label="nominal physics",
        arm_safety_radius_m=CONTRACT.task.arm_safety_radius_m,
    )
    # The pixel at the transformed end-effector position carries the
    # end-effector marker color (#D55E00).
    assert image.getpixel((round(px), round(py))) == (0xD5, 0x5E, 0x00)
