"""Compile the Typst report from staged, receipted bundle data.

Compilation is a pure rendering step: every displayed number lives
under ``report/generated/`` (placed there by ``stage_report`` after
bundle verification), and the Typst sources templated around it hold no
statistics of their own. The step fails closed on an unstaged
directory, an absent or contradictory receipt, an unknown content
status, or an unpinned Typst version, because the report PDF is a
public surface and must never be built from unverified inputs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from residual_worlds.paths import artifact_root, repository_root
from residual_worlds.release.schema import CONTENT_STATUSES
from residual_worlds.reporting.stage import RECEIPT_NAME, generated_dir

_TYPST_VERSION_PREFIX = "typst 0.15"


class ReportError(RuntimeError):
    pass


def build_report(output_path: Path | None = None) -> dict[str, Any]:
    """Compile ``report/main.typ`` against the staged bundle data."""
    generated = generated_dir()
    data_path = generated / "report_data.json"
    if not data_path.exists():
        raise ReportError(
            "report/generated/ is not staged; run stage_report on a verified bundle first"
        )
    report_data = json.loads(data_path.read_text(encoding="utf-8"))
    content_status = report_data.get("content_status")
    if content_status not in CONTENT_STATUSES:
        raise ReportError(f"staged report data has unknown content_status {content_status!r}")

    receipt_path = generated / RECEIPT_NAME
    if not receipt_path.exists():
        if content_status == "final":
            raise ReportError(
                "staged data claims content_status 'final' but no stage receipt exists; "
                "a final report compiles only from a receipted staging of a verified bundle"
            )
        raise ReportError(
            "report/generated/ lacks stage_receipt.json; restage the bundle"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("content_status") != content_status:
        raise ReportError("stage receipt and staged report data disagree on content_status")

    version = subprocess.run(
        ["typst", "--version"], capture_output=True, text=True, check=False
    )
    if version.returncode != 0 or not version.stdout.startswith(_TYPST_VERSION_PREFIX):
        raise ReportError(
            "typst is missing or not the pinned series: expected a version starting "
            f"with {_TYPST_VERSION_PREFIX!r}, got {version.stdout.strip()!r}"
        )

    root = repository_root()
    if output_path is None:
        output_path = artifact_root() / "reports" / "manual" / "residual_worlds_report.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiled = subprocess.run(
        [
            "typst",
            "compile",
            "--root",
            str(root),
            str(root / "report" / "main.typ"),
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if compiled.returncode != 0:
        raise ReportError("typst compilation failed:\n" + compiled.stderr)
    return {
        "pdf": str(output_path),
        # Typst does not report a count on success; the PDF is authoritative.
        "pages": None,
        "content_status": receipt["content_status"],
        "interpretation_state": receipt.get("interpretation_state"),
    }
