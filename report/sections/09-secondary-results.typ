#import "../lib.typ": *

= Secondary results

Secondary contrasts and the success-versus-budget curves are supporting
evidence: they carry Holm-adjusted p-values, are never promoted to
confirmatory claims, and only executed budgets appear --- nothing is
interpolated.

#let secondary = results.at("secondary", default: ())
#if secondary.len() == 0 [
  The staged bundle records no secondary contrasts; this section renders
  them from the bundle when they exist.
] else [
  #figure(
    table(
      columns: 4,
      align: (left, right, right, right),
      stroke: 0.5pt + neutral-grey.transparentize(50%),
      table.header(
        [Contrast], [Estimate], [95% interval], [Holm-adjusted p],
      ),
      ..secondary
        .map(row => (
          raw(row.contrast_id),
          fmt-pp(row.estimate),
          fmt-interval(row.lower_95, row.upper_95),
          if row.holm_adjusted_p == none { "---" } else {
            str(row.holm_adjusted_p)
          },
        ))
        .flatten(),
    ),
    caption: [Secondary contrasts, read from the staged bundle.],
  )
]

#bundle-figure("F04_budget_curve")
