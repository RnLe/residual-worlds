"""Artifact envelope: atomicity, immutability, corruption detection."""

from pathlib import Path

import pytest

from residual_worlds.provenance import (
    ArtifactError,
    is_complete,
    read_manifest,
    verify_artifact,
    write_artifact,
)


def _populate(directory: Path) -> None:
    (directory / "payload.txt").write_text("42\n", encoding="utf-8")
    (directory / "nested").mkdir()
    (directory / "nested" / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")


def test_write_and_verify_roundtrip(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_artifact(destination, "test", {"spec_id": "abc"}, {}, _populate)
    assert is_complete(destination)
    manifest = verify_artifact(destination)
    assert manifest["kind"] == "test"
    assert manifest["identities"]["spec_id"] == "abc"
    paths = {entry["path"] for entry in manifest["files"]}
    assert paths == {"payload.txt", "nested/table.csv"}


def test_refuses_overwrite_of_complete_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_artifact(destination, "test", {}, {}, _populate)
    with pytest.raises(ArtifactError):
        write_artifact(destination, "test", {}, {}, _populate)


def test_failed_populate_leaves_nothing(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"

    def explode(directory: Path) -> None:
        (directory / "partial.txt").write_text("x", encoding="utf-8")
        raise RuntimeError("simulated failure mid-write")

    with pytest.raises(RuntimeError):
        write_artifact(destination, "test", {}, {}, explode)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []  # no staging debris either


def test_tampering_is_detected(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_artifact(destination, "test", {}, {}, _populate)
    (destination / "payload.txt").write_text("43\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        verify_artifact(destination)


def test_added_file_is_detected(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_artifact(destination, "test", {}, {}, _populate)
    (destination / "smuggled.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ArtifactError, match="coverage mismatch"):
        verify_artifact(destination)


def test_missing_file_is_detected(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_artifact(destination, "test", {}, {}, _populate)
    (destination / "payload.txt").unlink()
    with pytest.raises(ArtifactError, match="coverage mismatch"):
        verify_artifact(destination)


def test_incomplete_directory_is_not_read(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()
    (destination / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not complete"):
        read_manifest(destination)


def test_extra_manifest_keys(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_artifact(
        destination, "test", {}, {}, _populate, extra_manifest={"termination_reason": "SUCCESS"}
    )
    assert read_manifest(destination)["termination_reason"] == "SUCCESS"
