#import "../lib.typ": *

= Model classes

#let swatch(key) = box(
  width: 0.85em,
  height: 0.85em,
  fill: method-colors.at(key),
  radius: 1.5pt,
  baseline: 12%,
)

Five conditions are compared. Names, colors, and marker cues below are
fixed across every figure and surface of the study; markers are the
non-color cue for the same identity.

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    stroke: 0.5pt + neutral-grey.transparentize(50%),
    table.header([Condition], [Cue], [Summary]),
    [#swatch("nominal") #method-label("nominal")],
    [#method-cues.at("nominal")],
    [The unchanged analytical model; no adaptation data. Measures how far
     the original approximate equations carry control in a changed world.],
    [#swatch("fitted_physics") #method-label("fitted_physics")],
    [#method-cues.at("fitted_physics")],
    [Five interpretable parameters (point payload, diagonal viscous
     damping, per-joint actuator gain) fitted by bounded deterministic
     optimization. Asks whether low-dimensional recalibration already
     explains the mismatch.],
    [#swatch("blackbox") #method-label("blackbox")],
    [#method-cues.at("blackbox")],
    [An MLP ensemble predicting the complete acceleration from data alone;
     the capacity-matched twin of the residual condition.],
    [#swatch("residual") #method-label("residual")],
    [#method-cues.at("residual")],
    [Nominal physics plus a learned acceleration correction. The
     correction network's final layer starts at exactly zero, so the
     untrained model reproduces nominal physics bit-for-bit; this zero
     initialization is the tested inductive bias.],
    [#swatch("oracle") #method-label("oracle")],
    [#method-cues.at("oracle")],
    [Plans on the exact target dynamics through the same approximate
     planner; a reference for planner-limited performance, not a
     performance ceiling.],
  ),
  caption: [Model conditions with their fixed display names, palette
    entries, and non-color marker cues.],
)

The two neural conditions use identical features, widths, activation,
output convention, and training procedure; their only structural
difference is whether the nominal acceleration is added to the network
output. Each neural condition is a small deterministic bootstrap ensemble.
The ensemble is a practical variance diagnostic and a mean-cost planning
device, never a calibrated posterior: the planner rolls every candidate
through each member separately and averages member costs, because a
nonlinear rollout of an averaged model is not the average of the member
rollouts.
