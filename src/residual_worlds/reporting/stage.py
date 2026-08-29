"""Stage report inputs from one verified public result bundle.

The Typst sources never open a bundle directly. Staging verifies the
bundle first (structure, checksums, status enums, figure-registry
coverage), then copies exactly the payloads the report may display into
``report/generated/`` and records every copied file with its SHA-256 in
a receipt. The compile step refuses to run without that receipt, so a
PDF is always traceable to one verified bundle. The staged directory is
deleted and recreated wholesale, never patched, so stale files from an
earlier bundle cannot survive a restage.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from residual_worlds.identity import file_sha256
from residual_worlds.paths import repository_root
from residual_worlds.release.verify_bundle import verify_bundle

RECEIPT_NAME = "stage_receipt.json"

# Bundle payload -> staged name. The key results travel under one
# neutral name so the Typst side reads a single fixed entry point.
_SUMMARY_COPIES = (
    ("summary/key_results.json", "report_data.json"),
    ("summary/study_summary.json", "study_summary.json"),
    ("summary/limitations.json", "limitations.json"),
)
_FIGURE_SUFFIXES = (".svg", ".png")


def generated_dir() -> Path:
    """The staged-report directory consumed by ``report/main.typ``."""
    return repository_root() / "report" / "generated"


def stage_report(bundle_directory: Path) -> dict[str, Any]:
    """Verify ``bundle_directory`` and stage its report payloads."""
    verification = verify_bundle(bundle_directory)

    destination = generated_dir()
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "figures").mkdir(parents=True)

    staged: dict[str, str] = {}

    def install(source: Path, relative: str) -> None:
        target = destination / relative
        shutil.copy2(source, target)
        staged[relative] = file_sha256(target)

    for bundle_relative, staged_name in _SUMMARY_COPIES:
        install(bundle_directory / bundle_relative, staged_name)
    install(
        bundle_directory / "figures/figure_registry.json",
        "figures/figure_registry.json",
    )
    for path in sorted((bundle_directory / "figures").iterdir()):
        if path.suffix.lower() in _FIGURE_SUFFIXES:
            install(path, f"figures/{path.name}")

    receipt = {
        "schema": 1,
        "source_bundle": str(bundle_directory.resolve()),
        "content_status": verification["content_status"],
        "interpretation_state": verification["interpretation_state"],
        "analysis_id": verification["analysis_id"],
        "files": staged,
    }
    (destination / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "generated": str(destination),
        "content_status": verification["content_status"],
        "interpretation_state": verification["interpretation_state"],
        "files": len(staged),
    }
