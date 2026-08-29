#import "../lib.typ": *

= Reproducibility appendix

This document is a pure rendering of one verified public result bundle;
it recomputes nothing.

- *Bundle identity.* Analysis #raw(results.analysis_id), protocol
  #raw(results.protocol_tag), content status
  #raw(results.content_status), interpretation state
  #raw(results.interpretation_state).
- *Staging.* The staging step verifies the bundle's manifest, checksums,
  status enums, and figure registry, deletes and recreates
  `report/generated/` wholesale, and writes `stage_receipt.json`
  recording the source bundle path and the SHA-256 of every staged file
  (#stage-receipt.files.len() files in the current receipt). Compilation
  refuses to run without a receipt.
- *Toolchain.* The Typst series is pinned in `report/typst-version.txt`;
  the build step rejects any other series and compiles with the project
  root as the file-system root, so the document can read staged files
  only. No network access occurs at build time.
- *Rebuild.* From the repository root:

```
uv run python -c "
from pathlib import Path
from residual_worlds.reporting.build import build_report
from residual_worlds.reporting.stage import stage_report
stage_report(Path('artifacts/public_result'))
build_report()
"
```

- *Statistics.* Estimates, intervals, and counts shown anywhere in this
  report originate in the analysis artifact referenced by the bundle;
  the staged JSON files are redacted copies, each traceable through the
  bundle manifest to its analysis source.
