"""Schema of the public result bundle.

One bundle directory feeds the README result block, the Typst report,
and the website; none of them may hold a second copy of scientific
truth. Two independent status fields travel with every bundle:

* ``content_status``: exactly one of ``schematic`` (watermarked
  placeholder data for development and CI), ``pilot`` (real machinery,
  non-protected data), or ``final`` (protected evaluation under a
  frozen protocol);
* ``interpretation_state``: the templated reading of the primary
  contrast (``no_results``, ``supports_primary_direction``,
  ``small_or_inconclusive``, ``opposite_direction``,
  ``protocol_deviation``).

They are independent: final evidence can be inconclusive. Verification
fails closed on any unknown value.
"""

from __future__ import annotations

CONTENT_STATUSES = ("schematic", "pilot", "final")
INTERPRETATION_STATES = (
    "no_results",
    "supports_primary_direction",
    "small_or_inconclusive",
    "opposite_direction",
    "protocol_deviation",
)

# Required bundle payloads, relative to the bundle root.
REQUIRED_FILES = (
    "summary/key_results.json",
    "summary/study_summary.json",
    "summary/limitations.json",
    "data/availability.json",
    "data/primary_effects.json",
    "data/budget_curve.json",
    "data/success_matrix.json",
    "data/failures.json",
    "figures/figure_registry.json",
)

KEY_RESULTS_REQUIRED = (
    "schema",
    "content_status",
    "interpretation_state",
    "protocol_tag",
    "analysis_id",
    "primary",
)

PRIMARY_REQUIRED = (
    "contrast_id",
    "estimate",
    "lower_95",
    "upper_95",
    "practical_threshold",
    "pipelines",
    "scenarios",
    "world_id",
    "budget",
)
