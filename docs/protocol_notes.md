# Protocol working notes

Running notes on decisions, open questions, and pre-freeze evidence.
Everything here is method-blind: observations come from nominal
dynamics, the exact-dynamics reference, and structural checks only,
never from fitted, black-box, or residual outcomes.

## Numerical verification (G1)

- Full analytic/energy/convergence suite passes; artifact under
  `artifacts/verification/` with energy and convergence plots.
- Provisional substep result: with valid-state sampling up to 80% of
  the speed limit, the smallest RK4 substep count meeting the one-step
  gate (p99 ≤ 1e-4 normalized) is **5 per 50 ms interval**
  (1 substep: p99 ≈ 7.5e-3; 2: ≈ 6.5e-4; 5: ≈ 1.6e-5). The G2 stress
  grid on the finally calibrated worlds decides the frozen count; the
  draft's `substeps_per_control_step: 1` will very likely not survive.
- float32 planning wrapper deviates from float64 truth by ≤ 6.8e-7
  normalized per step, far inside the gate.

## Controller/task calibration evidence for G2 (important)

Closed-loop probing with the **exact nominal model** under the draft
task values exposed three coupled pathologies. All three must be on the
table for the G2 `task_geometry_and_speed` / `task_cost` proposals,
because with the draft values the exact-dynamics reference cannot come
close to the required 38/40 calibration threshold.

1. **The elbow is torque-rich and inertia-poor.** Effective elbow
   inertia (det M / M11) is ≈ 0.02 kg·m²; the ±4 N·m command bound
   therefore spans ≈ ±200 rad/s². CEM exploration in latent space with
   sigma = 1 (typical |u| ≈ 3 N·m through tanh) crosses the 8 rad/s
   speed limit within 2–3 control steps, so at many states *every*
   sampled candidate is invalid and the executed final mean is
   arbitrary. Candidate remedies: per-joint torque bounds (e.g. elbow
   ≈ 1.5 N·m, which still dominates its ≤ 1.3 N·m gravity load),
   higher viscous damping, or per-joint latent scaling in the
   controller (a controller change, so it would have to be frozen with
   the profile).

2. **The dwell criterion demands ~0.4% torque precision.** Completing a
   target requires end-effector speed below 0.10 m/s for four
   consecutive steps. Holding the light elbow that still requires a
   sustained elbow-torque error below ≈ 0.08 N·m (about 2% of full
   scale) while the CEM standard-deviation floor alone injects ±0.2
   N·m of candidate jitter. In probing runs the exact-model controller
   hovered at the target for 80+ steps without ever completing a
   dwell. Raising the dwell speed threshold to 0.30 m/s made the same
   task complete in 8 steps. The threshold (or the torque scale) needs
   G2 attention.

3. **A large irrecoverable gravity region with no cost signal.** For
   shoulder angles below ≈ 1.1 rad the gravity torque exceeds the 4
   N·m bound, so a resting arm that drifts low cannot recover. The
   frozen cost has obstacle and joint-angle barriers but no velocity or
   recoverability term, so a finite-horizon planner happily approaches
   both the speed limit and the low-q1 region and gets trapped a few
   steps later. Scenario targets (y ≥ 0.18 m) mostly avoid the region,
   but starts near its boundary will fail for every method. Candidate
   remedies: a soft velocity barrier in the common cost, or restricting
   the scenario initial-state band.

The smoke configuration (`configs/smoke.yaml`) already uses a 0.30 m/s
dwell threshold and latent sigma 0.5 so the machinery can be exercised
end to end; the main contract keeps the draft values untouched until
the G2 proposals are made and receipted.

The first complete smoke run (46 control rows, all five methods, both
budgets) confirmed the point quantitatively: with the draft task-space
sampling every episode of every method, including the exact-dynamics
reference, terminated in `HARD_LIMIT_OR_SPEED`. The machinery closed
correctly (all rows structurally complete, the analysis produced its
templated `small_or_inconclusive` state on an all-zero success panel),
but no scientific data collection can start before the G2 calibration
loop resolves the three pathologies above. That is exactly the kind of
failure the calibration thresholds (38/40 exact-reference successes,
8–28/40 nominal) exist to catch.

## Scenario generation

- Twelve joint strata = six sector visit orders × obstacle chord (1st
  or 2nd). Seeded rotation balances banks exactly (12 → one per
  stratum; 24 → two; 40 → max spread of one).
- Structural feasibility uses analytic IK (both branches) plus a
  connected-component check on the joint-space grid graph with all
  clearance checks at ≥ 2 cm beyond the safety radius. Generation cost
  is ≈ 3–20 s per scenario depending on grid resolution; fully
  deterministic in (bank, index).

## Environment

- Torch's default CPU thread pool oversubscribes the shared machine
  catastrophically (67× slowdown measured on small geometry batches);
  the package caps it to 2 threads (`RW_TORCH_CPU_THREADS` overrides).
