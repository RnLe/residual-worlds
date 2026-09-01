// Arm, world, and schedule constants for the preview. They are copied from
// configs/experiment_contract.yaml (composite_standard) and from the
// Python media package; tests/fixtures/arm_golden.json pins every value,
// so a change on either side fails a test instead of drifting quietly.

export type Pair = readonly [number, number];

export interface ArmParams {
  readonly linkLengthsM: Pair;
  readonly comLengthsM: Pair;
  readonly massesKg: Pair;
  readonly inertiasKgM2: Pair;
  readonly viscousNmSRad: Pair;
  readonly gravityMS2: number;
  readonly torqueLimitNm: Pair;
  readonly qMinRad: Pair;
  readonly qMaxRad: Pair;
  readonly speedLimitRadS: Pair;
}

export interface FrictionParams {
  readonly viscousNmSRad: Pair;
  readonly coulombNm: Pair;
  readonly lowSpeedPeakNm: Pair;
  readonly stribeckVelocityRadS: Pair;
  readonly smoothingVelocityRadS: Pair;
}

export interface ActuatorParams {
  readonly gain: Pair;
  readonly deadzoneNm: Pair;
}

export interface WorldParams {
  readonly payloadKg: number | null;
  readonly friction: FrictionParams | null;
  readonly actuator: ActuatorParams | null;
  readonly elasticCouplingNm: number | null;
}

export interface ScheduleParams {
  readonly periodS: number;
  readonly qCenterRad: Pair;
  readonly amplitudeRad: Pair;
  readonly phaseRad: Pair;
  readonly harmonics: Pair;
  readonly kp: number;
  readonly kd: number;
  readonly warmupPeriods: number;
  readonly ghostSteps: number;
}

export const DT_S = 0.05;
export const SUBSTEPS = 1;

export const ARM: ArmParams = {
  linkLengthsM: [0.5, 0.5],
  comLengthsM: [0.25, 0.25],
  massesKg: [1.0, 1.0],
  inertiasKgM2: [0.0208333333333333, 0.0208333333333333],
  viscousNmSRad: [0.05, 0.05],
  gravityMS2: 9.81,
  torqueLimitNm: [4.0, 4.0],
  qMinRad: [-0.5235987756, -2.617993878],
  qMaxRad: [2.617993878, 2.617993878],
  speedLimitRadS: [8.0, 8.0],
};

export const WORLD: WorldParams = {
  payloadKg: 0.25,
  friction: {
    viscousNmSRad: [0.05, 0.05],
    coulombNm: [0.2, 0.12],
    lowSpeedPeakNm: [0.28, 0.18],
    stribeckVelocityRadS: [0.35, 0.3],
    smoothingVelocityRadS: [0.04, 0.04],
  },
  actuator: { gain: [0.86, 1.12], deadzoneNm: [0.12, 0.08] },
  elasticCouplingNm: null,
};

// The arm is weak against gravity, so the reference keeps its centre of
// mass over the base: the forearm folds against the shoulder lean.
export const SCHEDULE: ScheduleParams = {
  periodS: 7.0,
  qCenterRad: [1.856, -1.0],
  amplitudeRad: [0.3, 0.75],
  phaseRad: [0.0, Math.PI],
  harmonics: [1, 1],
  kp: 25.0,
  kd: 7.0,
  warmupPeriods: 2,
  ghostSteps: 6,
};
