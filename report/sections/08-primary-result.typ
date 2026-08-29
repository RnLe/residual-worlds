#import "../lib.typ": *

= Primary result

#status-banner

#if results.interpretation_state == "no_results" [
  #interpretation-sentence The planned confirmatory cell, read from the
  staged bundle, is contrast #raw(results.primary.contrast_id) in world
  #raw(results.primary.world_id) at a budget of #results.primary.budget
  transitions, over #results.primary.pipelines pipeline replicates
  $times$ #results.primary.scenarios scenario families, with a practical
  threshold of #fmt-pp(results.primary.practical_threshold). No estimate
  or interval is displayed because none is protected.
] else [
  #let primary = results.primary
  The preregistered contrast #raw(primary.contrast_id) estimates a paired
  success-rate difference of #fmt-pp(primary.estimate) with a 95%
  crossed-bootstrap interval of
  #fmt-interval(primary.lower_95, primary.upper_95), against a practical
  threshold of #fmt-pp(primary.practical_threshold). The estimate is
  computed over #primary.pipelines pipeline replicates $times$
  #primary.scenarios scenario families in world #raw(primary.world_id) at
  a budget of #primary.budget transitions.

  #interpretation-sentence
]

#bundle-figure("F03_primary_effect")
