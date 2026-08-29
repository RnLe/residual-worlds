#import "../lib.typ": *

= Diagnostics

Diagnostics are descriptive and never confirmatory; each is rendered from
the bundle and cross-referenced to its source table.

- *Open-loop prediction error.* Every eligible origin, registered before
  any model output existed, is rolled along the logged actions at the
  declared horizons. RMSE is conditional on finite origins and is always
  reported beside its numerator and denominator; the all-origin invalid
  fraction is reported separately, so a model is never rewarded for
  diverging on the hard cases.
- *Failure accounting.* Episode terminations are partitioned into
  successes, collisions, hard joint-limit violations, timeouts,
  non-finite states, and training failures; nothing is silently dropped,
  and a training-failed condition still yields a structurally complete
  record with no fabricated numbers.
- *Ensemble spread.* Member disagreement along evaluation rollouts is
  reported as a variance diagnostic; it is not calibrated uncertainty and
  is never used for decision rules.
- *Model--objective coupling.* Prediction error and closed-loop success
  are shown side by side per condition, since improvements in one need
  not transfer to the other @lambert2020objective.
