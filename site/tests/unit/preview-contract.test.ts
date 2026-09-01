// The preview is a pure function of its constants: two runs agree bit for
// bit, the loop stays inside the arm's limits, and under Node (no
// matchMedia) the motion helper reports reduced motion so the static path
// is the one tests exercise.

import { describe, expect, it } from "vitest";

import { motionOK } from "../../src/anim/raf";
import { ARM, SCHEDULE } from "../../src/arm/params";
import { simulateLoop } from "../../src/arm/schedule";

describe("preview loop", () => {
  it("is deterministic", () => {
    const a = simulateLoop();
    const b = simulateLoop();
    expect(a.states).toEqual(b.states);
    expect(a.ghosts).toEqual(b.ghosts);
  });

  it("stays inside joint, speed, and torque limits", () => {
    const loop = simulateLoop();
    expect(loop.states.length).toBe(Math.round(SCHEDULE.periodS / loop.dtS));
    for (const s of loop.states) {
      for (const j of [0, 1] as const) {
        expect(s[j]).toBeGreaterThanOrEqual(ARM.qMinRad[j]);
        expect(s[j]).toBeLessThanOrEqual(ARM.qMaxRad[j]);
        expect(Math.abs(s[2 + j] as number)).toBeLessThan(ARM.speedLimitRadS[j]);
      }
    }
    for (const a of loop.actions) {
      expect(Math.abs(a[0])).toBeLessThanOrEqual(ARM.torqueLimitNm[0]);
      expect(Math.abs(a[1])).toBeLessThanOrEqual(ARM.torqueLimitNm[1]);
    }
  });

  it("reports reduced motion without matchMedia", () => {
    expect(motionOK()).toBe(false);
  });
});
