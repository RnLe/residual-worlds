// The TypeScript physics must agree with the Python implementation. The
// golden numbers are written by `residual-worlds fixture-arm-golden`;
// parameters are compared exactly, evaluations to tight tolerances.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { forwardKinematics } from "../../src/arm/kinematics";
import { nominalAcceleration, type Vec2, type Vec4 } from "../../src/arm/nominal";
import { rk4Transition } from "../../src/arm/integrate";
import { ARM, DT_S, SCHEDULE, SUBSTEPS, WORLD } from "../../src/arm/params";
import { reference, simulateLoop, trackingTorque } from "../../src/arm/schedule";
import { targetAcceleration } from "../../src/arm/world";

const GOLDEN_PATH = fileURLToPath(new URL("../../../tests/fixtures/arm_golden.json", import.meta.url));

interface Sample {
  state: Vec4;
  action: Vec2;
  elbow: Vec2;
  hand: Vec2;
  nominal_acc: Vec2;
  target_acc: Vec2;
  tracking_torque: Vec2;
}

interface Rollout {
  state: Vec4;
  action: Vec2;
  nominal: Vec4[];
  target: Vec4[];
}

interface Golden {
  dt_s: number;
  substeps: number;
  arm: Record<string, number | [number, number]>;
  world: {
    payload_kg: number | null;
    friction: Record<string, [number, number]> | null;
    actuator: Record<string, [number, number]> | null;
    elastic_coupling_nm: number | null;
  };
  schedule: Record<string, number | [number, number]>;
  reference_at_1p234_s: { q: Vec2; qd: Vec2; qdd: Vec2 };
  samples: Sample[];
  rollouts: Rollout[];
  loop: {
    frames: number;
    states_first: Vec4;
    states_last: Vec4;
    actions_first: Vec2;
    ghost_first_end: Vec4;
    residual_abs_max: Vec2;
  };
}

const golden = JSON.parse(readFileSync(GOLDEN_PATH, "utf8")) as Golden;

function expectClose(actual: readonly number[], expected: readonly number[], tolerance: number): void {
  expect(actual.length).toBe(expected.length);
  actual.forEach((value, i) => {
    expect(Math.abs(value - (expected[i] as number))).toBeLessThanOrEqual(tolerance);
  });
}

describe("parameters", () => {
  it("arm constants equal the contract values", () => {
    expect(ARM.linkLengthsM).toEqual(golden.arm.link_lengths_m);
    expect(ARM.comLengthsM).toEqual(golden.arm.com_lengths_m);
    expect(ARM.massesKg).toEqual(golden.arm.masses_kg);
    expect(ARM.inertiasKgM2).toEqual(golden.arm.inertias_kg_m2);
    expect(ARM.viscousNmSRad).toEqual(golden.arm.viscous_nm_s_rad);
    expect(ARM.gravityMS2).toBe(golden.arm.gravity_m_s2);
    expect(ARM.torqueLimitNm).toEqual(golden.arm.torque_limit_nm);
    expect(ARM.qMinRad).toEqual(golden.arm.q_min_rad);
    expect(ARM.qMaxRad).toEqual(golden.arm.q_max_rad);
    expect(ARM.speedLimitRadS).toEqual(golden.arm.speed_limit_rad_s);
    expect(DT_S).toBe(golden.dt_s);
    expect(SUBSTEPS).toBe(golden.substeps);
  });

  it("world constants equal the resolved composite world", () => {
    expect(WORLD.payloadKg).toBe(golden.world.payload_kg);
    expect(WORLD.elasticCouplingNm).toBe(golden.world.elastic_coupling_nm);
    const friction = golden.world.friction;
    expect(friction).not.toBeNull();
    if (friction !== null && WORLD.friction !== null) {
      expect(WORLD.friction.viscousNmSRad).toEqual(friction.viscous_nm_s_rad);
      expect(WORLD.friction.coulombNm).toEqual(friction.coulomb_nm);
      expect(WORLD.friction.lowSpeedPeakNm).toEqual(friction.low_speed_peak_nm);
      expect(WORLD.friction.stribeckVelocityRadS).toEqual(friction.stribeck_velocity_rad_s);
      expect(WORLD.friction.smoothingVelocityRadS).toEqual(friction.smoothing_velocity_rad_s);
    }
    const actuator = golden.world.actuator;
    expect(actuator).not.toBeNull();
    if (actuator !== null && WORLD.actuator !== null) {
      expect(WORLD.actuator.gain).toEqual(actuator.gain);
      expect(WORLD.actuator.deadzoneNm).toEqual(actuator.deadzone_nm);
    }
  });

  it("schedule constants equal the Python schedule", () => {
    expect(SCHEDULE.periodS).toBe(golden.schedule.period_s);
    expect(SCHEDULE.qCenterRad).toEqual(golden.schedule.q_center_rad);
    expect(SCHEDULE.amplitudeRad).toEqual(golden.schedule.amplitude_rad);
    expect(SCHEDULE.phaseRad).toEqual(golden.schedule.phase_rad);
    expect(SCHEDULE.harmonics).toEqual(golden.schedule.harmonics);
    expect(SCHEDULE.kp).toBe(golden.schedule.kp);
    expect(SCHEDULE.kd).toBe(golden.schedule.kd);
    expect(SCHEDULE.warmupPeriods).toBe(golden.schedule.warmup_periods);
    expect(SCHEDULE.ghostSteps).toBe(golden.schedule.ghost_steps);
  });
});

describe("evaluations", () => {
  it("reference trajectory", () => {
    const ref = reference(1.234);
    expectClose(ref.q, golden.reference_at_1p234_s.q, 1e-12);
    expectClose(ref.qd, golden.reference_at_1p234_s.qd, 1e-12);
    expectClose(ref.qdd, golden.reference_at_1p234_s.qdd, 1e-12);
  });

  it("kinematics, accelerations, and tracking torque at every sample", () => {
    golden.samples.forEach((sample, i) => {
      const pose = forwardKinematics(sample.state[0], sample.state[1], ARM);
      expectClose(pose.elbow, sample.elbow, 1e-12);
      expectClose(pose.hand, sample.hand, 1e-12);
      expectClose(nominalAcceleration(sample.state, sample.action, ARM), sample.nominal_acc, 1e-9);
      expectClose(targetAcceleration(sample.state, sample.action, WORLD, ARM), sample.target_acc, 1e-9);
      expectClose(trackingTorque(sample.state, 0.37 * i), sample.tracking_torque, 1e-9);
    });
  });

  it("RK4 rollouts under nominal and target dynamics", () => {
    for (const rollout of golden.rollouts) {
      let nominal: Vec4 = rollout.state;
      let target: Vec4 = rollout.state;
      rollout.nominal.forEach((expected, i) => {
        if (i > 0) {
          nominal = rk4Transition((s, a) => nominalAcceleration(s, a, ARM), nominal, rollout.action, DT_S, SUBSTEPS);
          target = rk4Transition((s, a) => targetAcceleration(s, a, WORLD, ARM), target, rollout.action, DT_S, SUBSTEPS);
        }
        expectClose(nominal, expected, 1e-8);
        expectClose(target, rollout.target[i] as Vec4, 1e-8);
      });
    }
  });

  it("the recorded loop matches its Python twin", () => {
    const loop = simulateLoop();
    expect(loop.states.length).toBe(golden.loop.frames);
    expectClose(loop.states[0] as Vec4, golden.loop.states_first, 1e-6);
    expectClose(loop.states[loop.states.length - 1] as Vec4, golden.loop.states_last, 1e-6);
    expectClose(loop.actions[0] as Vec2, golden.loop.actions_first, 1e-6);
    const ghost = loop.ghosts[0] as readonly Vec4[];
    expectClose(ghost[ghost.length - 1] as Vec4, golden.loop.ghost_first_end, 1e-6);
    const absMax: Vec2 = [0, 0];
    for (const r of loop.residual) {
      absMax[0] = Math.max(absMax[0], Math.abs(r[0]));
      absMax[1] = Math.max(absMax[1], Math.abs(r[1]));
    }
    expectClose(absMax, golden.loop.residual_abs_max, 1e-6);
  });
});
