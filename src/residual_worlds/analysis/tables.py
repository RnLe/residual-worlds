"""Machine-readable result tables exported from one analysis artifact.

Every aggregate table keeps its provenance: the analysis identifier
travels with each export, and rows retain the identities needed to walk
back to individual evaluation artifacts. CSVs are for humans and the
website's exact-data views; the parquet files inside the analysis
artifact remain the numerical source.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _write_csv(path: Path, columns: dict[str, list[Any]]) -> None:
    keys = list(columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for row_index in range(len(columns[keys[0]])):
            writer.writerow([columns[key][row_index] for key in keys])


def export_tables(analysis_directory: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []

    mapping = {
        "T03_contrasts.csv": "contrasts.parquet",
        "T04_success_and_failures.csv": "method_summaries.parquet",
        "T10_evidence_rows.csv": "cell_outcomes.parquet",
    }
    optional = {"T05_prediction_metrics.csv": "prediction_summaries.parquet"}
    for csv_name, parquet_name in {**mapping, **optional}.items():
        source = analysis_directory / parquet_name
        if not source.exists():
            continue
        columns = pq.read_table(source).to_pydict()
        _write_csv(output_dir / csv_name, columns)
        exported.append(csv_name)

    interpretation = json.loads(
        (analysis_directory / "interpretation.json").read_text(encoding="utf-8")
    )
    (output_dir / "interpretation.json").write_text(
        json.dumps(interpretation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    exported.append("interpretation.json")
    return {"exported": exported, "directory": str(output_dir)}
