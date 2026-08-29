#import "../lib.typ": *

= Confirmatory design and statistics

*Primary estimand.* The primary contrast is the paired closed-loop
success-rate difference between the residual and black-box conditions at
the primary world and budget. With $Y_(r s)^((m)) in {0, 1}$ the task
success of method $m$ under pipeline replicate $r$ and scenario family
$s$, over $R$ pipelines and $S$ scenario families,

$ hat(Delta) = 1 / (R S) sum_(r = 1)^(R) sum_(s = 1)^(S)
  ( Y_(r s)^(("res")) - Y_(r s)^(("bb")) ). $

The replication structure is explicit throughout: pipelines and scenario
families are crossed factors, and the $R times S$ episode rows are never
treated as $R S$ independent experiments.

*Uncertainty.* Interval estimates come from a crossed stratified
bootstrap: pipelines are resampled with replacement, scenarios are
resampled with replacement within each structural stratum (preserving the
designed per-stratum composition), and method pairing inside every
selected cell is preserved exactly.

*Decision rule.* The confirmatory reading combines the 95% interval with
a practical-relevance threshold fixed before any protected evaluation; the
threshold value is displayed beside the primary estimate and is read from
the bundle, never restated in the sources. An exact sign-flip enumeration
over pipeline effects is reported as an assumption-dependent sensitivity
check --- it requires sign symmetry of the paired pipeline effects, which
method labels were never randomized to guarantee --- and never replaces
the interval-plus-threshold rule. Secondary contrasts carry Holm-adjusted
p-values and are interpreted as supporting evidence only.

*Templated interpretation.* The verbal reading of the primary contrast is
restricted to five preregistered template sentences keyed by the bundle's
interpretation state; this report renders the recorded sentence verbatim
and adds no free-form gloss.
