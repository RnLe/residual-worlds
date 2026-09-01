// Planar two-link forward kinematics: q1 from the positive x axis, q2
// relative to link 1, gravity along negative y.

import type { ArmParams } from "./params";

export interface Pose {
  readonly elbow: readonly [number, number];
  readonly hand: readonly [number, number];
}

export function forwardKinematics(q1: number, q2: number, arm: ArmParams): Pose {
  const [l1, l2] = arm.linkLengthsM;
  const elbowX = l1 * Math.cos(q1);
  const elbowY = l1 * Math.sin(q1);
  return {
    elbow: [elbowX, elbowY],
    hand: [elbowX + l2 * Math.cos(q1 + q2), elbowY + l2 * Math.sin(q1 + q2)],
  };
}
