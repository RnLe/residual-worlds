"""Render the animated preview: what the model imagined against what happened.

Frame k draws the true arm at step k and, faded, the nominal model's
rollout that was launched ``ghost_steps`` earlier from the then-true
state under the torques actually issued. The two hands land in different
places; that gap, per joint and in acceleration terms, is traced below
as the residual. Colors and type follow the site tokens so the image
sits naturally on the page.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.patches import Arc, Circle  # noqa: E402
from PIL import Image  # noqa: E402

from residual_worlds.config import ExperimentContract  # noqa: E402
from residual_worlds.media.loop import SCHEDULE, Loop, Schedule, simulate_loop  # noqa: E402
from residual_worlds.types import ArmParameters  # noqa: E402

PAPER = "#FBF9F3"
INK = "#3E4A54"
INK_SOFT = "#5F6C77"
LINE = "#DDD3BE"
STEEL = "#4D7B9E"
STEEL_LIGHT = "#A5C6DF"
STARK = "#EBA538"
RESIDUAL = "#7E57C2"
RESIDUAL_DEEP = "#6A46A8"

SIZE_LIMIT_BYTES = 3 * 1024 * 1024


@dataclass(frozen=True)
class RenderOptions:
    width: int = 960
    height: int = 540
    fps: int = 20
    webp_quality: int = 80
    gif_stride: int = 1  # write every n-th frame to the GIF to hold its size


def _fk(q: np.ndarray, arm: ArmParameters) -> tuple[np.ndarray, np.ndarray]:
    l1, l2 = arm.link_lengths_m
    elbow = np.array([l1 * np.cos(q[0]), l1 * np.sin(q[0])])
    hand = elbow + np.array([l2 * np.cos(q[0] + q[1]), l2 * np.sin(q[0] + q[1])])
    return elbow, hand


class _Frame:
    """One figure whose artists are updated per frame (no re-layout)."""

    def __init__(self, loop: Loop, arm: ArmParameters, schedule: Schedule, opt: RenderOptions):
        self.loop = loop
        self.arm = arm
        self.schedule = schedule
        self.opt = opt
        w, h = opt.width, opt.height
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=PAPER)
        self.canvas = FigureCanvasAgg(self.fig)

        # Arm panel: pixel coordinates, y up, base fixed.
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.set_axis_off()
        self.ax.set_facecolor(PAPER)
        self.scale = 0.57 * h
        self.base = np.array([0.63 * w, 0.31 * h])

        reach = sum(arm.link_lengths_m) * self.scale
        self.ax.add_patch(
            Arc(
                tuple(self.base),
                2 * reach,
                2 * reach,
                theta1=0,
                theta2=180,
                ec=LINE,
                lw=1.2,
                ls=(0, (4, 6)),
            )
        )
        self.ax.plot(
            [self.base[0] - 24, self.base[0] + 24],
            [self.base[1], self.base[1]],
            color=LINE,
            lw=2.5,
            solid_capstyle="round",
        )

        (self.ghost_path,) = self.ax.plot([], [], color=STEEL, lw=1.8, ls=(0, (3, 4)))
        (self.ghost_upper,) = self.ax.plot([], [], color=STEEL_LIGHT, lw=11, solid_capstyle="round")
        (self.ghost_lower,) = self.ax.plot([], [], color=STEEL_LIGHT, lw=11, solid_capstyle="round")
        self.ghost_hand = Circle((0, 0), 9, fill=True, fc=PAPER, ec=STEEL, lw=2.5)
        self.ax.add_patch(self.ghost_hand)

        (self.gap,) = self.ax.plot([], [], color=RESIDUAL, lw=2.6)
        (self.trail,) = self.ax.plot([], [], color=STARK, lw=2.2, alpha=0.8)
        (self.upper,) = self.ax.plot([], [], color=INK, lw=11, solid_capstyle="round")
        (self.lower,) = self.ax.plot([], [], color=INK, lw=11, solid_capstyle="round")
        self.joint_base = Circle(tuple(self.base), 8, fc=INK, ec=PAPER, lw=2)
        self.joint_elbow = Circle((0, 0), 7, fc=INK, ec=PAPER, lw=2)
        self.hand = Circle((0, 0), 9, fc=STARK, ec=INK, lw=2)
        for patch in (self.joint_base, self.joint_elbow, self.hand):
            self.ax.add_patch(patch)

        ahead_s = schedule.ghost_steps * loop.dt_s
        x0, y0 = 0.035 * w, h - 0.075 * h
        self.ax.text(
            x0, y0, "the model imagined", color=STEEL, fontsize=19, fontweight="bold", va="center"
        )
        self.ax.text(
            x0,
            y0 - 30,
            f"nominal physics, {ahead_s:.1f} s ago",
            color=INK_SOFT,
            fontsize=14,
            va="center",
        )
        self.ax.text(
            x0, y0 - 80, "the world did", color=INK, fontsize=19, fontweight="bold", va="center"
        )
        self.ax.text(
            x0, y0 - 110, "payload, friction, actuator", color=INK_SOFT, fontsize=14, va="center"
        )
        self.gap_label = self.ax.text(0, 0, "", color=RESIDUAL, fontsize=14, fontweight="bold")
        self.clock = self.ax.text(
            w - 0.035 * w, h - 0.075 * h, "", color=INK_SOFT, fontsize=14, ha="right", va="center"
        )

        # Residual strip along the bottom.
        left, right = 0.035 * w, 0.965 * w
        bottom, top = 0.045 * h, 0.20 * h
        self.strip = (left, right, bottom, top)
        self.ax.plot([left, right], [bottom, bottom], color=LINE, lw=1)
        self.ax.text(
            left,
            top + 10,
            "the residual",
            color=RESIDUAL,
            fontsize=19,
            fontweight="bold",
            va="bottom",
        )
        self.ax.text(
            right,
            top + 10,
            "acceleration the equations got wrong: shoulder (solid), elbow (dashed)",
            color=INK_SOFT,
            fontsize=14,
            ha="right",
            va="bottom",
        )
        self.limit = float(max(np.abs(loop.residual).max(), 1e-9))
        mid = 0.5 * (bottom + top)
        self.ax.plot([left, right], [mid, mid], color=LINE, lw=1, ls=(0, (2, 4)))
        (self.res_1,) = self.ax.plot([], [], color=RESIDUAL, lw=2.2)
        (self.res_2,) = self.ax.plot([], [], color=RESIDUAL_DEEP, lw=2.2, ls=(0, (5, 3)))
        (self.cursor,) = self.ax.plot([], [], color=INK_SOFT, lw=1)
        self.fills: list[Any] = []

    def _px(self, xy: np.ndarray) -> np.ndarray:
        return self.base + xy * self.scale

    def draw(self, k: int) -> Image.Image:
        loop, arm, sch = self.loop, self.arm, self.schedule
        n, h_steps = loop.frames, sch.ghost_steps
        launch = (k - h_steps) % n
        ghost = loop.ghosts[launch]

        g_elbow, g_hand = (self._px(p) for p in _fk(ghost[-1, :2], arm))
        elbow, hand = (self._px(p) for p in _fk(loop.states[k, :2], arm))

        ghost_hands = np.array([self._px(_fk(gs[:2], arm)[1]) for gs in ghost])
        self.ghost_path.set_data(ghost_hands[:, 0], ghost_hands[:, 1])
        self.ghost_upper.set_data([self.base[0], g_elbow[0]], [self.base[1], g_elbow[1]])
        self.ghost_lower.set_data([g_elbow[0], g_hand[0]], [g_elbow[1], g_hand[1]])
        self.ghost_hand.center = (g_hand[0], g_hand[1])

        idx = [(launch + i) % n for i in range(h_steps + 1)]
        trail = np.array([self._px(_fk(loop.states[i, :2], arm)[1]) for i in idx])
        self.trail.set_data(trail[:, 0], trail[:, 1])

        self.gap.set_data([g_hand[0], hand[0]], [g_hand[1], hand[1]])
        mid = 0.5 * (g_hand + hand)
        along = hand - g_hand
        norm = float(np.hypot(along[0], along[1])) or 1.0
        perp = np.array([-along[1], along[0]]) / norm
        if perp[1] < 0:
            perp = -perp
        self.gap_label.set_position((mid[0] + 16 * perp[0] + 8, mid[1] + 16 * perp[1] + 4))
        self.gap_label.set_text("gap")

        self.upper.set_data([self.base[0], elbow[0]], [self.base[1], elbow[1]])
        self.lower.set_data([elbow[0], hand[0]], [elbow[1], hand[1]])
        self.joint_elbow.center = (elbow[0], elbow[1])
        self.hand.center = (hand[0], hand[1])
        self.clock.set_text(f"t = {k * loop.dt_s:.1f} s")

        left, right, bottom, top = self.strip
        xs = left + (right - left) * np.arange(n) / (n - 1)
        centre = 0.5 * (bottom + top)
        half = 0.5 * (top - bottom) * 0.92
        upto = k + 1
        y1 = centre + half * loop.residual[:upto, 0] / self.limit
        y2 = centre + half * loop.residual[:upto, 1] / self.limit
        for fill in self.fills:
            fill.remove()
        self.fills = [
            self.ax.fill_between(xs[:upto], centre, y2, color=RESIDUAL_DEEP, alpha=0.10, lw=0),
            self.ax.fill_between(xs[:upto], centre, y1, color=RESIDUAL, alpha=0.18, lw=0),
        ]
        self.res_1.set_data(xs[:upto], y1)
        self.res_2.set_data(xs[:upto], y2)
        self.cursor.set_data([xs[k], xs[k]], [bottom, top])

        self.canvas.draw()  # type: ignore[no-untyped-call]
        rgba = np.asarray(self.canvas.buffer_rgba())  # type: ignore[no-untyped-call]
        return Image.fromarray(rgba[:, :, :3].copy(), "RGB")

    def close(self) -> None:
        plt.close(self.fig)


def render_frames(
    loop: Loop,
    arm: ArmParameters,
    schedule: Schedule = SCHEDULE,
    opt: RenderOptions | None = None,
) -> list[Image.Image]:
    frame = _Frame(loop, arm, schedule, opt or RenderOptions())
    try:
        return [frame.draw(k) for k in range(loop.frames)]
    finally:
        frame.close()


def write_animations(
    frames: list[Image.Image], output_dir: Path, opt: RenderOptions | None = None
) -> dict[str, Any]:
    opt = opt or RenderOptions()
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000 / opt.fps))
    webp = output_dir / "preview.webp"
    frames[0].save(
        webp,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=opt.webp_quality,
        method=6,
    )
    gif = output_dir / "preview.gif"
    gif_frames = frames[:: opt.gif_stride]
    gif_frames[0].save(
        gif,
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms * opt.gif_stride,
        loop=0,
        optimize=True,
    )
    sizes = {"webp_bytes": webp.stat().st_size, "gif_bytes": gif.stat().st_size}
    for name, size in sizes.items():
        if size > SIZE_LIMIT_BYTES:
            raise RuntimeError(f"{name} is {size} bytes, above the {SIZE_LIMIT_BYTES} byte limit")
    return {"webp": str(webp), "gif": str(gif), "frames": len(frames), **sizes}


def render_preview(
    contract: ExperimentContract,
    output_dir: Path,
    world_id: str = "composite_standard",
    opt: RenderOptions | None = None,
) -> dict[str, Any]:
    opt = opt or RenderOptions()
    loop = simulate_loop(contract, world_id, SCHEDULE)
    frames = render_frames(loop, contract.arm, SCHEDULE, opt)
    return {"world_id": world_id, **write_animations(frames, output_dir, opt)}
