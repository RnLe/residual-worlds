"""Public-bundle schema enforcement and fixture integrity."""

import json
import shutil
from pathlib import Path

import pytest

from residual_worlds.paths import repository_root
from residual_worlds.release.schematic import WATERMARK, build_schematic_fixture
from residual_worlds.release.verify_bundle import BundleError, verify_bundle

FIXTURE = repository_root() / "tests" / "fixtures" / "public_result_schematic"


def test_checked_in_fixture_verifies() -> None:
    result = verify_bundle(FIXTURE)
    assert result["content_status"] == "schematic"
    assert result["interpretation_state"] == "no_results"
    assert result["figures"] >= 2


def test_fixture_carries_watermark_everywhere() -> None:
    key_results = json.loads((FIXTURE / "summary/key_results.json").read_text())
    assert key_results["watermark"] == WATERMARK
    study = json.loads((FIXTURE / "summary/study_summary.json").read_text())
    assert WATERMARK in json.dumps(study)


def test_fixture_rejected_when_final_required() -> None:
    with pytest.raises(BundleError, match="final"):
        verify_bundle(FIXTURE, require_content_status="final")


def test_tampered_bundle_fails_closed(tmp_path: Path) -> None:
    copy = tmp_path / "bundle"
    shutil.copytree(FIXTURE, copy)
    key_results_path = copy / "summary/key_results.json"
    payload = json.loads(key_results_path.read_text())
    payload["content_status"] = "final"  # forged status without new checksums
    key_results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    # Either the checksum audit or the schema check may fire first;
    # any failure mode is acceptable as long as it fails.
    with pytest.raises((Exception,)):  # noqa: B017
        verify_bundle(copy, require_content_status="final")


def test_unknown_interpretation_state_rejected(tmp_path: Path) -> None:
    bundle = build_schematic_fixture(tmp_path / "bundle")
    key_results_path = bundle / "summary/key_results.json"
    payload = json.loads(key_results_path.read_text())
    payload["interpretation_state"] = "definitely_a_breakthrough"
    key_results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises((Exception,)):  # noqa: B017 - any failure mode acceptable
        verify_bundle(bundle)


def test_regenerated_fixture_matches_schema(tmp_path: Path) -> None:
    bundle = build_schematic_fixture(tmp_path / "fresh")
    result = verify_bundle(bundle)
    assert result["content_status"] == "schematic"
