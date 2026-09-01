// The preview episode, mirrored from the Python media package: a
// computed-torque tracker on the exact target dynamics follows a smooth
// reference for one period, and the nominal model is rolled forward from
// every recorded state under the same torques.

import { rk4Transition } from "./integrate";
import { nominalAcceleration, type Vec2, type Vec4 } from "./nominal";
import {
  ARM,
  DT_S,
  SCHEDULE,
  SUBSTEPS,
  WORLD,
  type ArmParams,
  type Pair,
  type ScheduleParams,
  type WorldParams,
} from "./params";
import { elasticTorque, targetAcceleration, worldTerms } from "./world";

export interface Reference {
  readonly q: Vec2;
  readonly qd: Vec2;
  readonly qdd: Vec2;
}

export function reference(t: number, schedule: ScheduleParams = SCHEDULE): Reference {
  const q: Vec2 = [0, 0];
  const qd: Vec2 = [0, 0];
  const qdd: Vec2 = [0, 0];
  for (const j of [0, 1] as const) {
    const omega = (2.0 * Math.PI * schedule.harmonics[j]) / schedule.periodS;
    const arg = omega * t + schedule.phaseRad[j];
    const amplitude = schedule.amplitudeRad[j];
    q[j] = schedule.qCenterRad[j] + amplitude * Math.sin(arg);
    qd[j] = amplitude * omega * Math.cos(arg);
    qdd[j] = -amplitude * omega * omega * Math.sin(arg);
  }
  return { q, qd, qdd };
}

export function trackingTorque(
  state: Vec4,
  t: number,
  arm: ArmParams = ARM,
  world: WorldParams = WORLD,
  schedule: ScheduleParams = SCHEDULE,
): Vec2 {
  const ref = reference(t, schedule);
  const q: Pair = [state[0], state[1]];
  const qd: Pair = [state[2], state[3]];
  const command: Vec2 = [
    ref.qdd[0] + schedule.kp * (ref.q[0] - q[0]) + schedule.kd * (ref.qd[0] - qd[0]),
    ref.qdd[1] + schedule.kp * (ref.q[1] - q[1]) + schedule.kd * (ref.qd[1] - qd[1]),
  ];
  const terms = worldTerms(q, qd, world, arm);
  const [m11, m12, m22] = terms.mass;
  const applied: Vec2 = [
    m11 * command[0] + m12 * command[1] + terms.coriolis[0] + terms.gravity[0] + terms.friction[0],
    m12 * command[0] + m22 * command[1] + terms.coriolis[1] + terms.gravity[1] + terms.friction[1],
  ];
  if (world.elasticCouplingNm !== null) {
    const e = elasticTorque(q, world.elasticCouplingNm);
    applied[0] -= e[0];
    applied[1] -= e[1];
  }
  const u: Vec2 = [0, 0];
  for (const j of [0, 1] as const) {
    let value = applied[j];
    if (world.actuator !== null) {
      value =
        Math.sign(applied[j]) *
        (Math.abs(applied[j]) / world.actuator.gain[j] + world.actuator.deadzoneNm[j]);
    }
    const limit = arm.torqueLimitNm[j];
    u[j] = Math.min(Math.max(value, -limit), limit);
  }
  return u;
}

export interface Loop {
  readonly dtS: number;
  readonly states: readonly Vec4[];
  readonly actions: readonly Vec2[];
  readonly nominalAcc: readonly Vec2[];
  readonly targetAcc: readonly Vec2[];
  readonly residual: readonly Vec2[];
  readonly ghosts: readonly (readonly Vec4[])[];
}

export function simulateLoop(
  arm: ArmParams = ARM,
  world: WorldParams = WORLD,
  schedule: ScheduleParams = SCHEDULE,
  dt: number = DT_S,
  substeps: number = SUBSTEPS,
): Loop {
  const steps = Math.round(schedule.periodS / dt);
  const trueAcc = (s: Vec4, a: Pair): Vec2 => targetAcceleration(s, a, world, arm);
  const nominalAcc = (s: Vec4, a: Pair): Vec2 => nominalAcceleration(s, a, arm);

  const start = reference(0.0, schedule);
  let state: Vec4 = [start.q[0], start.q[1], start.qd[0], start.qd[1]];
  for (let k = 0; k < schedule.warmupPeriods * steps; k += 1) {
    const action = trackingTorque(state, k * dt, arm, world, schedule);
    state = rk4Transition(trueAcc, state, action, dt, substeps);
  }

  const states: Vec4[] = [];
  const actions: Vec2[] = [];
  const t0 = schedule.warmupPeriods * schedule.periodS;
  for (let k = 0; k < steps; k += 1) {
    states.push(state);
    const action = trackingTorque(state, t0 + k * dt, arm, world, schedule);
    actions.push(action);
    state = rk4Transition(trueAcc, state, action, dt, substeps);
  }

  const nominalAccs = states.map((s, k) => nominalAcc(s, actions[k] as Vec2));
  const targetAccs = states.map((s, k) => trueAcc(s, actions[k] as Vec2));
  const residual = nominalAccs.map((n, k) => {
    const t = targetAccs[k] as Vec2;
    return [t[0] - n[0], t[1] - n[1]] as Vec2;
  });

  const ghosts = states.map((s, k) => {
    const path: Vec4[] = [s];
    let ghost = s;
    for (let i = 0; i < schedule.ghostSteps; i += 1) {
      ghost = rk4Transition(nominalAcc, ghost, actions[(k + i) % steps] as Vec2, dt, substeps);
      path.push(ghost);
    }
    return path;
  });

  return { dtS: dt, states, actions, nominalAcc: nominalAccs, targetAcc: targetAccs, residual, ghosts };
}
