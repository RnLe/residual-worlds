// Nominal rigid-body dynamics, term by term as in physics/nominal.py:
//
//     M0(q) qdd + c0(q, qd) + g0(q) + B0 qd = u.
//
// A symmetric 2x2 inertia matrix is stored as [m11, m12, m22].

import type { ArmParams, Pair } from "./params";

export type Vec2 = [number, number];
export type Vec4 = [number, number, number, number];
export type Sym2 = [number, number, number];

export function massMatrix(q: Pair, arm: ArmParams): Sym2 {
  const [l1] = arm.linkLengthsM;
  const [lc1, lc2] = arm.comLengthsM;
  const [m1, m2] = arm.massesKg;
  const [i1, i2] = arm.inertiasKgM2;
  const cosQ2 = Math.cos(q[1]);
  const m11 = i1 + i2 + m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2.0 * l1 * lc2 * cosQ2);
  const m12 = i2 + m2 * (lc2 ** 2 + l1 * lc2 * cosQ2);
  const m22 = i2 + m2 * lc2 ** 2;
  return [m11, m12, m22];
}

export function coriolisVector(q: Pair, qd: Pair, arm: ArmParams): Vec2 {
  const [l1] = arm.linkLengthsM;
  const lc2 = arm.comLengthsM[1];
  const m2 = arm.massesKg[1];
  const h = -(m2 * l1 * lc2) * Math.sin(q[1]);
  const [qd1, qd2] = qd;
  return [h * (2.0 * qd1 * qd2 + qd2 ** 2), -h * qd1 ** 2];
}

export function gravityVector(q: Pair, arm: ArmParams): Vec2 {
  const [l1] = arm.linkLengthsM;
  const [lc1, lc2] = arm.comLengthsM;
  const [m1, m2] = arm.massesKg;
  const g = arm.gravityMS2;
  const shared = m2 * lc2 * g * Math.cos(q[0] + q[1]);
  return [(m1 * lc1 + m2 * l1) * g * Math.cos(q[0]) + shared, shared];
}

export function dampingTorque(qd: Pair, arm: ArmParams): Vec2 {
  return [arm.viscousNmSRad[0] * qd[0], arm.viscousNmSRad[1] * qd[1]];
}

// Solve M x = rhs for the symmetric 2x2 case without forming an inverse.
export function solve2(m: Sym2, rhs: Pair): Vec2 {
  const [m11, m12, m22] = m;
  const det = m11 * m22 - m12 * m12;
  return [(m22 * rhs[0] - m12 * rhs[1]) / det, (m11 * rhs[1] - m12 * rhs[0]) / det];
}

export function nominalAcceleration(state: Vec4, action: Pair, arm: ArmParams): Vec2 {
  const q: Pair = [state[0], state[1]];
  const qd: Pair = [state[2], state[3]];
  const c = coriolisVector(q, qd, arm);
  const g = gravityVector(q, arm);
  const b = dampingTorque(qd, arm);
  return solve2(massMatrix(q, arm), [
    action[0] - c[0] - g[0] - b[0],
    action[1] - c[1] - g[1] - b[1],
  ]);
}
