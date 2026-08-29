// Shared data access, palette, and formatting for the report.
//
// Every number displayed anywhere in the report is read here from the
// staged bundle data under generated/; section files carry prose and
// structure only, never literal statistics. Unknown status values
// abort compilation instead of rendering a best guess.

#let results = json("generated/report_data.json")
#let study = json("generated/study_summary.json")
#let limitations-data = json("generated/limitations.json")
#let figure-registry = json("generated/figures/figure_registry.json")
#let stage-receipt = json("generated/stage_receipt.json")

// ------------------------------------------------------------- statuses
#let content-statuses = ("schematic", "pilot", "final")
#if not content-statuses.contains(results.content_status) {
  panic("unknown content_status: " + results.content_status)
}

// The reading of the primary contrast is restricted to these fixed
// template sentences; no free-form gloss of a result is ever generated.
#let interpretation-sentences = (
  no_results: "No protected results exist yet; the primary test is planned.",
  supports_primary_direction: "The preregistered primary contrast met both the interval and the practical-threshold rule.",
  small_or_inconclusive: "The primary estimate was small or its interval was inconclusive.",
  opposite_direction: "The primary contrast resolved in the opposite direction.",
  protocol_deviation: "A protocol deviation occurred; see the report.",
)
#let interpretation-sentence = interpretation-sentences.at(
  results.interpretation_state,
  default: none,
)
#if interpretation-sentence == none {
  panic("unknown interpretation_state: " + results.interpretation_state)
}

// -------------------------------------------------------------- palette
// Matches the figure palette in src/residual_worlds/analysis/figures.py;
// every colored mark carries a non-color cue alongside.
#let page-background = rgb("#F7F7F3")
#let ink = rgb("#202124")
#let neutral-grey = rgb("#6B7280")
#let accent-warn = rgb("#A65F00")

#let method-colors = (
  truth: rgb("#202124"),
  nominal: rgb("#A65F00"),
  fitted_physics: rgb("#009E73"),
  blackbox: rgb("#0072B2"),
  residual: rgb("#7E57C2"),
  oracle: rgb("#6B7280"),
)
#let method-names = (
  nominal: "nominal physics",
  fitted_physics: "fitted physics",
  blackbox: "black box",
  residual: "residual",
  oracle: "exact-dynamics reference",
)
#let method-cues = (
  nominal: "circle marker",
  fitted_physics: "square marker",
  blackbox: "triangle marker",
  residual: "diamond marker",
  oracle: "dashed line, x marker",
)
#let method-label(key) = text(fill: method-colors.at(key), method-names.at(key))

// ----------------------------------------------------------- formatting
// Success-rate differences are reported in percentage points.
#let as-pp(x) = str(calc.round(x * 100, digits: 1))
#let fmt-pp(x) = as-pp(x) + " pp"
#let fmt-interval(lo, hi) = "[" + as-pp(lo) + ", " + as-pp(hi) + "] pp"

// ------------------------------------------------------- status surface
#let watermark-text = "SCHEMATIC DATA - NOT RESULTS"
#let status-note = (
  schematic: watermark-text,
  pilot: "PILOT DATA - NOT PROTECTED RESULTS",
  final: none,
).at(results.content_status)

#let page-watermark = if results.content_status == "schematic" {
  // The wide box keeps the rotated line unbroken across the page diagonal.
  place(
    center + horizon,
    rotate(
      -54deg,
      box(
        width: 34cm,
        align(
          center,
          text(
            size: 40pt,
            weight: "bold",
            fill: neutral-grey.transparentize(60%),
            watermark-text,
          ),
        ),
      ),
    ),
  )
}

#let page-header = text(size: 8pt, fill: neutral-grey)[
  Residual Worlds --- content status: #results.content_status
  #h(1fr)
  #if status-note != none { text(fill: accent-warn, weight: "bold", status-note) }
]

#let status-styles = (
  schematic: (fill: rgb("#F6E8D5"), stroke: accent-warn),
  pilot: (fill: rgb("#DFEBF4"), stroke: rgb("#0072B2")),
  final: (fill: rgb("#DFF0E9"), stroke: rgb("#009E73")),
)
#let status-banner = {
  let style = status-styles.at(results.content_status)
  block(
    width: 100%,
    inset: 10pt,
    radius: 3pt,
    fill: style.fill,
    stroke: 1pt + style.stroke,
  )[
    #text(weight: "bold", size: 11pt)[Content status: #upper(results.content_status)]
    #h(1fr)
    #text(size: 9pt, fill: neutral-grey)[protocol #raw(results.protocol_tag)]
    \
    #interpretation-sentence
  ]
}

// -------------------------------------------------------------- figures
#let figure-not-staged(figure-id) = block(
  width: 100%,
  inset: 14pt,
  stroke: (paint: neutral-grey, dash: "dashed"),
)[
  #text(fill: neutral-grey)[
    Figure #raw(figure-id) is not staged. Restage from a verified bundle
    that provides it.
  ]
]

// Renders a registry figure by identifier, or an explicit box when the
// identifier, its vector file, or its staged copy is absent.
#let bundle-figure(figure-id) = {
  let matches = figure-registry.filter(entry => entry.figure_id == figure-id)
  if matches.len() == 0 {
    return figure-not-staged(figure-id)
  }
  let entry = matches.first()
  let vector-files = entry.files.filter(name => name.ends-with(".svg"))
  if vector-files.len() == 0 {
    return figure-not-staged(figure-id)
  }
  let relative = "figures/" + vector-files.first()
  if relative not in stage-receipt.files {
    return figure-not-staged(figure-id)
  }
  let badge = if "badge" in entry {
    h(0.6em)
    box(
      inset: (x: 4pt, y: 2pt),
      stroke: 0.5pt + neutral-grey,
      radius: 2pt,
      baseline: 25%,
      text(size: 8pt, fill: neutral-grey, entry.badge),
    )
  }
  figure(
    image("generated/" + relative, width: 92%),
    caption: [#entry.caption#badge],
  )
}
