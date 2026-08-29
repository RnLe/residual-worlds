// Residual Worlds --- result report.
//
// Compiled with Typst (series pinned in typst-version.txt) by
// residual_worlds.reporting.build, which requires a receipted staging
// of one verified public result bundle. Every displayed number is read
// from generated/; the sources hold no statistics of their own.

#import "lib.typ": *

#let results = json("generated/report_data.json")

#set document(title: study.title)
#set page(
  paper: "a4",
  fill: page-background,
  margin: (x: 2.4cm, top: 2.8cm, bottom: 2.6cm),
  numbering: "1",
  header: page-header,
  background: page-watermark,
)
#set text(size: 10.5pt, fill: ink, lang: "en")
#set par(justify: true)
#set heading(numbering: "1.1")
#show figure.caption: set text(size: 9pt)
#show link: set text(fill: rgb("#0072B2"))

// ------------------------------------------------------------------ cover
#align(center)[
  #v(1.6cm)
  #text(size: 24pt, weight: "bold")[#study.title]
  #v(0.3cm)
  #text(size: 13pt, style: "italic", fill: neutral-grey)[#study.tagline]
  #v(1.0cm)
]

#status-banner

#if results.content_status == "schematic" [
  #v(0.4cm)
  #align(center)[
    #block(width: 88%, inset: 12pt, stroke: 1.5pt + accent-warn, radius: 3pt)[
      #text(weight: "bold", fill: accent-warn)[#watermark-text] \
      Every number, figure, and table in this document is placeholder
      layout data for exercising the reporting pipeline. Nothing in it
      is a finding.
    ]
  ]
]

#v(0.8cm)
*Study question.* #study.question

#v(0.5cm)
#text(size: 9pt, fill: neutral-grey)[
  Analysis #raw(results.analysis_id) --- protocol #raw(results.protocol_tag)
  --- content status #raw(results.content_status) --- interpretation state
  #raw(results.interpretation_state).
]

#pagebreak()
#outline(depth: 2)
#pagebreak()

// --------------------------------------------------------------- sections
#include "sections/01-abstract.typ"
#include "sections/02-motivation.typ"
#include "sections/03-dynamics.typ"
#include "sections/04-model-classes.typ"
#include "sections/05-data-protocol.typ"
#include "sections/06-controller.typ"
#include "sections/07-confirmatory-design.typ"
#include "sections/08-primary-result.typ"
#include "sections/09-secondary-results.typ"
#include "sections/10-diagnostics.typ"
#include "sections/11-limitations.typ"
#include "sections/12-reproducibility.typ"

#pagebreak()
#bibliography("bibliography.bib")
