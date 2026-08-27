"""Deterministic rendering of saved arm states.

Rendering is communication only: no policy consumes pixels, and drawing
never touches simulation state. Frames are produced with PIL from saved
trajectories through one versioned world-to-pixel transform shared by
every consumer (video frames, posters, site replays), so a position in
a video can be cross-checked against the same transform elsewhere.

Transform version 1: y up, arm base at (width/2, 0.72 * height),
uniform scale of ``height / 2.4`` pixels per meter (the +/-1 m reach
plus margin fits vertically).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw

from residual_worlds.physics.kinematics import link_segments
from residual_worlds.types import ArmParameters, Scenario

TRANSFORM_VERSION = 1

BACKGROUND = "#F7F7F3"
ARM_COLOR = "#202124"
OBSTACLE_COLOR = "#C94F4F"
OBSTACLE_MARGIN_COLOR = "#E5B9B9"
TARGET_PENDING = "#B8B8B2"
TARGET_ACTIVE = "#0072B2"
TARGET_DONE = "#009E73"
END_EFFECTOR_COLOR = "#D55E00"
TEXT_COLOR = "#3A3A38"


@dataclass(frozen=True)
class WorldToPixel:
    """Versioned affine world-to-pixel transform (version 1)."""

    width: int
    height: int

    @property
    def scale(self) -> float:
        return self.height / 2.4

    @property
    def base_xy(self) -> tuple[float, float]:
        return (self.width / 2.0, 0.72 * self.height)

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        bx, by = self.base_xy
        return (bx + x * self.scale, by - y * self.scale)


def render_state(
    q: torch.Tensor,
    scenario: Scenario,
    arm: ArmParameters,
    width: int,
    height: int,
    target_index: int,
    method_label: str,
    arm_safety_radius_m: float,
) -> Image.Image:
    """Render one configuration to a PIL image (deterministic)."""
    transform = WorldToPixel(width=width, height=height)
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    scale = transform.scale

    ox, oy, orad = scenario.obstacle_xy_radius_m
    cx, cy = transform.to_pixel(ox, oy)
    margin = (orad + arm_safety_radius_m) * scale
    draw.ellipse(
        (cx - margin, cy - margin, cx + margin, cy + margin),
        outline=OBSTACLE_MARGIN_COLOR,
        width=2,
    )
    body = orad * scale
    draw.ellipse((cx - body, cy - body, cx + body, cy + body), fill=OBSTACLE_COLOR)

    ring = 0.04 * scale
    for index, (tx, ty) in enumerate(scenario.targets_xy_m):
        px, py = transform.to_pixel(tx, ty)
        if index < target_index:
            color = TARGET_DONE
        elif index == target_index:
            color = TARGET_ACTIVE
        else:
            color = TARGET_PENDING
        draw.ellipse((px - ring, py - ring, px + ring, py + ring), outline=color, width=4)
        draw.text((px + ring + 3, py - 7), str(index + 1), fill=color)

    base, elbow, tip = link_segments(q.to(torch.float64), arm)
    points = [
        transform.to_pixel(float(p[0]), float(p[1])) for p in (base, elbow, tip)
    ]
    line_width = max(3, int(0.045 * scale))
    draw.line([points[0], points[1]], fill=ARM_COLOR, width=line_width)
    draw.line([points[1], points[2]], fill=ARM_COLOR, width=line_width)
    joint_radius = max(3, int(0.02 * scale))
    for px, py in points[:2]:
        draw.ellipse(
            (px - joint_radius, py - joint_radius, px + joint_radius, py + joint_radius),
            fill=ARM_COLOR,
        )
    px, py = points[2]
    ee = max(4, int(0.025 * scale))
    draw.ellipse((px - ee, py - ee, px + ee, py + ee), fill=END_EFFECTOR_COLOR)

    draw.text((10, 8), "SIMULATION", fill=TEXT_COLOR)
    draw.text((10, 24), method_label, fill=TEXT_COLOR)
    return image
