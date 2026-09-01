// The target world: nominal physics plus payload, Stribeck friction, and
// an actuator with gain and dead zone, composed exactly as in
// physics/components.py and physics/target.py.

import type { ArmParams, FrictionParams, Pair, WorldParams } from "./params";
import {
  coriolisVector,
  dampingTorque,
  gravityVector,
  massMatrix,
  solve2,
  type Sym2,
  type Vec2,
  type Vec4,
} from "./nominal";

export function payloadMassMatrix(q: Pair, payloadKg: number, arm: ArmParams): Sym2 {
  const [l1, l2] = arm.linkLengthsM;
  const cosQ2 = Math.cos(q[1]);
  return [
    payloadKg * (l1 ** 2 + l2 ** 2 + 2.0 * l1 * l2 * cosQ2),
    payloadKg * (l2 ** 2 + l1 * l2 * cosQ2),
    payloadKg * l2 ** 2,
  ];
}

export function payloadCoriolis(q: Pair, qd: Pair, payloadKg: number, arm: ArmParams): Vec2 {
  const [l1, l2] = arm.linkLengthsM;
  const h = -payloadKg * l1 * l2 * Math.sin(q[1]);
  const [qd1, qd2] = qd;
  return [h * (2.0 * qd1 * qd2 + qd2 ** 2), -h * qd1 ** 2];
}

export function payloadGravity(q: Pair, payloadKg: number, arm: ArmParams): Vec2 {
  const [l1, l2] = arm.linkLengthsM;
  const g = arm.gravityMS2;
  const shared = payloadKg * g * l2 * Math.cos(q[0] + q[1]);
  return [payloadKg * g * l1 * Math.cos(q[0]) + shared, shared];
}

// Per joint: b qd + [f_c + (f_s - f_c) exp(-(qd / v_s)^2)] tanh(qd / eps).
export function frictionTorque(qd: Pair, friction: FrictionParams): Vec2 {
  const out: Vec2 = [0, 0];
  for (const j of [0, 1] as const) {
    const b = friction.viscousNmSRad[j];
    const fc = friction.coulombNm[j];
    const fs = friction.lowSpeedPeakNm[j];
    const vs = friction.stribeckVelocityRadS[j];
    const eps = friction.smoothingVelocityRadS[j];
    const stribeck = fc + (fs - fc) * Math.exp(-((qd[j] / vs) ** 2));
    out[j] = b * qd[j] + stribeck * Math.tanh(qd[j] / eps);
  }
  return out;
}

// Gain, dead zone, then the physical clip.
export function appliedTorque(u: Pair, world: WorldParams, arm: ArmParams): Vec2 {
  if (world.actuator === null) return [u[0], u[1]];
  const out: Vec2 = [0, 0];
  for (const j of [0, 1] as const) {
    const gain = world.actuator.gain[j];
    const deadzone = world.actuator.deadzoneNm[j];
    const limit = arm.torqueLimitNm[j];
    const magnitude = Math.max(Math.abs(u[j]) - deadzone, 0.0);
    out[j] = Math.min(Math.max(gain * Math.sign(u[j]) * magnitude, -limit), limit);
  }
  return out;
}

export function elasticTorque(q: Pair, kc: number): Vec2 {
  const s = Math.sin(q[0] - q[1]);
  return [-kc * s, kc * s];
}

export interface WorldTerms {
  readonly mass: Sym2;
  readonly coriolis: Vec2;
  readonly gravity: Vec2;
  readonly friction: Vec2;
}

// Left-hand-side terms of the composed world at one state.
export function worldTerms(q: Pair, qd: Pair, world: WorldParams, arm: ArmParams): WorldTerms {
  let mass = massMatrix(q, arm);
  let coriolis = coriolisVector(q, qd, arm);
  let gravity = gravityVector(q, arm);
  if (world.payloadKg !== null) {
    const pm = payloadMassMatrix(q, world.payloadKg, arm);
    const pc = payloadCoriolis(q, qd, world.payloadKg, arm);
    const pg = payloadGravity(q, world.payloadKg, arm);
    mass = [mass[0] + pm[0], mass[1] + pm[1], mass[2] + pm[2]];
    coriolis = [coriolis[0] + pc[0], coriolis[1] + pc[1]];
    gravity = [gravity[0] + pg[0], gravity[1] + pg[1]];
  }
  const friction =
    world.friction !== null ? frictionTorque(qd, world.friction) : dampingTorque(qd, arm);
  return { mass, coriolis, gravity, friction };
}

export function targetAcceleration(
  state: Vec4,
  action: Pair,
  world: WorldParams,
  arm: ArmParams,
): Vec2 {
  const q: Pair = [state[0], state[1]];
  const qd: Pair = [state[2], state[3]];
  const terms = worldTerms(q, qd, world, arm);
  const right = appliedTorque(action, world, arm);
  if (world.elasticCouplingNm !== null) {
    const e = elasticTorque(q, world.elasticCouplingNm);
    right[0] += e[0];
    right[1] += e[1];
  }
  return solve2(terms.mass, [
    right[0] - terms.coriolis[0] - terms.gravity[0] - terms.friction[0],
    right[1] - terms.coriolis[1] - terms.gravity[1] - terms.friction[1],
  ]);
}
