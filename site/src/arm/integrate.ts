// Fixed-step RK4 over one control interval with the action held constant,
// the same scheme as physics/integrators.py.

import type { Pair } from "./params";
import type { Vec2, Vec4 } from "./nominal";

export type AccelerationFn = (state: Vec4, action: Pair) => Vec2;

function derivative(accel: AccelerationFn, s: Vec4, a: Pair): Vec4 {
  const qdd = accel(s, a);
  return [s[2], s[3], qdd[0], qdd[1]];
}

function axpy(s: Vec4, k: Vec4, h: number): Vec4 {
  return [s[0] + h * k[0], s[1] + h * k[1], s[2] + h * k[2], s[3] + h * k[3]];
}

export function rk4Transition(
  accel: AccelerationFn,
  state: Vec4,
  action: Pair,
  dt: number,
  substeps: number,
): Vec4 {
  const h = dt / substeps;
  let current = state;
  for (let i = 0; i < substeps; i += 1) {
    const k1 = derivative(accel, current, action);
    const k2 = derivative(accel, axpy(current, k1, 0.5 * h), action);
    const k3 = derivative(accel, axpy(current, k2, 0.5 * h), action);
    const k4 = derivative(accel, axpy(current, k3, h), action);
    current = [
      current[0] + (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
      current[1] + (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
      current[2] + (h / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
      current[3] + (h / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]),
    ];
  }
  return current;
}
