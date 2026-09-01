// The preview, drawn live: the true arm at step k, and faded, the nominal
// model's rollout launched ghostSteps earlier from the then-true state
// under the torques actually issued. The layout matches the offline
// render in the Python media package, so the static image and the live
// canvas look the same. Autoplay only without a reduced-motion
// preference; stepping buttons always work.

import { motionOK } from "./anim/raf";
import { forwardKinematics } from "./arm/kinematics";
import { ARM, SCHEDULE } from "./arm/params";
import { simulateLoop, type Loop } from "./arm/schedule";
import { el } from "./dom";

const W = 960;
const H = 540;

interface Palette {
  paper: string;
  ink: string;
  inkSoft: string;
  line: string;
  steel: string;
  steelLight: string;
  stark: string;
  residual: string;
  residualDeep: string;
}

function readPalette(): Palette {
  const style = getComputedStyle(document.documentElement);
  const token = (name: string, fallback: string): string => {
    const value = style.getPropertyValue(name).trim();
    return value === "" ? fallback : value;
  };
  return {
    paper: token("--paper-raised", "#fbf9f3"),
    ink: token("--ink-strong", "#3e4a54"),
    inkSoft: token("--ink-soft", "#5f6c77"),
    line: token("--line", "#ddd3be"),
    steel: token("--steel", "#4d7b9e"),
    steelLight: token("--steel-light", "#a5c6df"),
    stark: token("--stark", "#eba538"),
    residual: token("--m-residual", "#7e57c2"),
    residualDeep: token("--m-residual-deep", "#6a46a8"),
  };
}

const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

class Scene {
  private readonly base = { x: 0.63 * W, y: H - 0.31 * H };
  private readonly scale = 0.57 * H;
  private readonly strip = { left: 0.035 * W, right: 0.965 * W, top: H - 0.2 * H, bottom: H - 0.045 * H };
  private readonly limit: number;

  constructor(
    private readonly ctx: CanvasRenderingContext2D,
    private readonly loop: Loop,
    private readonly colors: Palette,
  ) {
    let limit = 1e-9;
    for (const r of loop.residual) limit = Math.max(limit, Math.abs(r[0]), Math.abs(r[1]));
    this.limit = limit;
  }

  private px(xy: readonly [number, number]): [number, number] {
    return [this.base.x + xy[0] * this.scale, this.base.y - xy[1] * this.scale];
  }

  private line(a: [number, number], b: [number, number], color: string, width: number, dash: number[] = []): void {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.restore();
  }

  private dot(c: [number, number], r: number, fill: string, stroke: string, width: number): void {
    const { ctx } = this;
    ctx.beginPath();
    ctx.arc(c[0], c[1], r, 0, 2 * Math.PI);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.lineWidth = width;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }

  private text(s: string, x: number, y: number, color: string, size: number, weight = "400", align: CanvasTextAlign = "left"): void {
    const { ctx } = this;
    ctx.fillStyle = color;
    ctx.font = `${weight} ${size}px ${FONT}`;
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    ctx.fillText(s, x, y);
  }

  draw(k: number): void {
    const { ctx, loop, colors } = this;
    const n = loop.states.length;
    const steps = SCHEDULE.ghostSteps;
    const launch = (((k - steps) % n) + n) % n;
    const ghost = loop.ghosts[launch] as readonly [number, number, number, number][];
    const ghostEnd = ghost[ghost.length - 1] as [number, number, number, number];
    const state = loop.states[k] as [number, number, number, number];

    ctx.fillStyle = colors.paper;
    ctx.fillRect(0, 0, W, H);

    // Reach arc and ground mark.
    const reach = (ARM.linkLengthsM[0] + ARM.linkLengthsM[1]) * this.scale;
    ctx.save();
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.arc(this.base.x, this.base.y, reach, Math.PI, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
    this.line([this.base.x - 24, this.base.y], [this.base.x + 24, this.base.y], colors.line, 2.5);

    // The ghost: imagined hand path, faded arm, hollow hand.
    const ghostPose = forwardKinematics(ghostEnd[0], ghostEnd[1], ARM);
    const gElbow = this.px(ghostPose.elbow);
    const gHand = this.px(ghostPose.hand);
    ctx.save();
    ctx.strokeStyle = colors.steel;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ghost.forEach((gs, i) => {
      const p = this.px(forwardKinematics(gs[0], gs[1], ARM).hand);
      if (i === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.stroke();
    ctx.restore();
    this.line([this.base.x, this.base.y], gElbow, colors.steelLight, 11);
    this.line(gElbow, gHand, colors.steelLight, 11);
    this.dot(gHand, 9, colors.paper, colors.steel, 2.5);

    // The true arm and its recent hand trail.
    const pose = forwardKinematics(state[0], state[1], ARM);
    const elbow = this.px(pose.elbow);
    const hand = this.px(pose.hand);
    ctx.save();
    ctx.strokeStyle = colors.stark;
    ctx.globalAlpha = 0.8;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i <= steps; i += 1) {
      const s = loop.states[(launch + i) % n] as [number, number, number, number];
      const p = this.px(forwardKinematics(s[0], s[1], ARM).hand);
      if (i === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    }
    ctx.stroke();
    ctx.restore();

    this.line(gHand, hand, colors.residual, 2.6);
    this.line([this.base.x, this.base.y], elbow, colors.ink, 11);
    this.line(elbow, hand, colors.ink, 11);
    this.dot([this.base.x, this.base.y], 8, colors.ink, colors.paper, 2);
    this.dot(elbow, 7, colors.ink, colors.paper, 2);
    this.dot(hand, 9, colors.stark, colors.ink, 2);

    const mid: [number, number] = [0.5 * (gHand[0] + hand[0]), 0.5 * (gHand[1] + hand[1])];
    const alongX = hand[0] - gHand[0];
    const alongY = hand[1] - gHand[1];
    const norm = Math.hypot(alongX, alongY) || 1;
    let perpX = -alongY / norm;
    let perpY = alongX / norm;
    if (perpY > 0) {
      perpX = -perpX;
      perpY = -perpY;
    }
    this.text("gap", mid[0] + 16 * perpX + 8, mid[1] + 16 * perpY - 4, colors.residual, 14, "700");

    // Labels.
    const x0 = 0.035 * W;
    const y0 = 0.075 * H;
    const aheadS = (steps * loop.dtS).toFixed(1);
    this.text("the model imagined", x0, y0, colors.steel, 19, "700");
    this.text(`nominal physics, ${aheadS} s ago`, x0, y0 + 30, colors.inkSoft, 14);
    this.text("the world did", x0, y0 + 80, colors.ink, 19, "700");
    this.text("payload, friction, actuator", x0, y0 + 110, colors.inkSoft, 14);
    this.text(`t = ${(k * loop.dtS).toFixed(1)} s`, W - 0.035 * W, y0, colors.inkSoft, 14, "400", "right");

    // Residual strip.
    const { left, right, top, bottom } = this.strip;
    const centre = 0.5 * (top + bottom);
    const half = 0.5 * (bottom - top) * 0.92;
    this.line([left, bottom], [right, bottom], colors.line, 1);
    this.line([left, centre], [right, centre], colors.line, 1, [2, 4]);
    this.text("the residual", left, top - 22, colors.residual, 19, "700");
    this.text(
      "acceleration the equations got wrong: shoulder (solid), elbow (dashed)",
      right,
      top - 22,
      colors.inkSoft,
      14,
      "400",
      "right",
    );
    const xs = (i: number): number => left + ((right - left) * i) / (n - 1);
    const ys = (v: number): number => centre - (half * v) / this.limit;
    const traces: { joint: 0 | 1; color: string; alpha: number; dash: number[] }[] = [
      { joint: 1, color: colors.residualDeep, alpha: 0.1, dash: [5, 3] },
      { joint: 0, color: colors.residual, alpha: 0.18, dash: [] },
    ];
    for (const { joint, color, alpha, dash } of traces) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(xs(0), centre);
      for (let i = 0; i <= k; i += 1) ctx.lineTo(xs(i), ys((loop.residual[i] as [number, number])[joint]));
      ctx.lineTo(xs(k), centre);
      ctx.closePath();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.fill();
      ctx.restore();
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i <= k; i += 1) {
        const p: [number, number] = [xs(i), ys((loop.residual[i] as [number, number])[joint])];
        if (i === 0) ctx.moveTo(p[0], p[1]);
        else ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
      ctx.restore();
    }
    this.line([xs(k), top], [xs(k), bottom], colors.inkSoft, 1);
  }
}

export function mountPreview(host: HTMLElement): void {
  const loop = simulateLoop();
  const canvas = el("canvas", "preview-canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.setAttribute("role", "img");
  canvas.setAttribute(
    "aria-label",
    "Two-link arm in its true world, with the nominal model's prediction from a moment " +
      "earlier drawn faded; the gap between the two hands is the model mismatch, and a " +
      "strip below traces the acceleration residual over time.",
  );
  const ctx = canvas.getContext("2d");
  if (ctx === null) throw new Error("canvas 2d context unavailable");
  const scene = new Scene(ctx, loop, readPalette());

  const n = loop.states.length;
  let frame = 0;
  let playing = false;
  let timer: number | null = null;

  const readout = el("p", "interactive-readout");
  const show = (k: number): void => {
    frame = ((k % n) + n) % n;
    scene.draw(frame);
    readout.textContent = `step ${frame + 1} of ${n}, t = ${(frame * loop.dtS).toFixed(2)} s`;
  };

  const controls = el("div", "control-row");
  const button = (label: string, onClick: () => void): HTMLButtonElement => {
    const b = el("button", "control-button", label);
    b.type = "button";
    b.addEventListener("click", onClick);
    controls.append(b);
    return b;
  };

  const stop = (): void => {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
    playing = false;
    playButton.textContent = "play";
    playButton.setAttribute("aria-pressed", "false");
  };
  const start = (): void => {
    if (playing) return;
    playing = true;
    playButton.textContent = "pause";
    playButton.setAttribute("aria-pressed", "true");
    timer = window.setInterval(() => show(frame + 1), loop.dtS * 1000);
  };

  const playButton = button("play", () => (playing ? stop() : start()));
  button("previous", () => {
    stop();
    show(frame - 1);
  });
  button("next", () => {
    stop();
    show(frame + 1);
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
  });

  show(0);
  const caption = host.querySelector("figcaption");
  const stack = el("div", "interactive-stack");
  stack.append(canvas, controls, readout);
  host.querySelector("img")?.remove();
  if (caption !== null) host.insertBefore(stack, caption);
  else host.append(stack);
  if (motionOK()) start();
}
