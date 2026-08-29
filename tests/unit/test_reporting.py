"""Report staging discipline and pinned Typst compilation."""

import json
import shutil
from pathlib import Path

import pytest

from residual_worlds.identity import file_sha256
from residual_worlds.paths import repository_root
from residual_worlds.reporting.build import ReportError, build_report
from residual_worlds.reporting.stage import stage_report

FIXTURE = repository_root() / "tests" / "fixtures" / "public_result_schematic"
GENERATED = repository_root() / "report" / "generated"


def test_build_without_staging_raises() -> None:
    shutil.rmtree(GENERATED, ignore_errors=True)
    with pytest.raises(ReportError, match="not staged"):
        build_report()


@pytest.mark.slow
def test_stage_and_build_pdf(tmp_path: Path) -> None:
    staged = stage_report(FIXTURE)
    assert staged["content_status"] == "schematic"
    assert (GENERATED / "report_data.json").exists()
    assert (GENERATED / "figures" / "figure_registry.json").exists()
    receipt = json.loads(
        (GENERATED / "stage_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["content_status"] == "schematic"
    assert receipt["interpretation_state"] == "no_results"
    assert receipt["files"]["report_data.json"] == file_sha256(
        GENERATED / "report_data.json"
    )

    output = tmp_path / "residual_worlds_report.pdf"
    built = build_report(output_path=output)
    assert built["content_status"] == "schematic"
    assert Path(built["pdf"]) == output
    assert output.exists()
    assert output.stat().st_size > 20_000
