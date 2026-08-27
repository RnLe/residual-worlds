"""Immutable artifact envelope: manifest, checksums, completion sentinel.

Every generated artifact directory follows the same discipline:

1. payload files are written into a temporary sibling directory;
2. ``manifest.json`` records the artifact kind, its identities, its
   input references, and a sorted ledger of every payload file with
   byte size and SHA-256;
3. ``checksums.sha256`` covers every installed file including the
   manifest (excluding itself and the sentinel);
4. an empty ``COMPLETE`` sentinel is written last, and the directory is
   atomically renamed into place.

A directory without ``COMPLETE`` is an aborted attempt and is never
read. A completed directory is never overwritten; rerunning a job with
the same identity either verifies the existing artifact or fails. This
is deliberately simple -- a single researcher machine does not need
distributed locking, but it does need to know that no result file was
half-written or silently edited after the fact.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from residual_worlds.identity import file_sha256

MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "checksums.sha256"
COMPLETE_NAME = "COMPLETE"
_RESERVED = {MANIFEST_NAME, CHECKSUMS_NAME, COMPLETE_NAME}


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, incomplete, or corrupted."""


def is_complete(directory: Path) -> bool:
    return (directory / COMPLETE_NAME).exists()


def _payload_ledger(directory: Path) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        if path.is_symlink():
            raise ArtifactError(f"symlinks are not permitted in artifacts: {path}")
        relative = path.relative_to(directory).as_posix()
        if relative in _RESERVED:
            continue
        ledger.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return ledger


def _write_checksums(directory: Path) -> None:
    lines: list[str] = []
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in {CHECKSUMS_NAME, COMPLETE_NAME}:
            continue
        lines.append(f"{file_sha256(path)}  {relative}")
    (directory / CHECKSUMS_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifact(
    destination: Path,
    kind: str,
    identities: dict[str, Any],
    inputs: dict[str, Any],
    populate: Callable[[Path], None],
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    """Create one immutable artifact directory at ``destination``.

    ``populate`` receives the temporary directory and writes all payload
    files. ``identities`` carries content-derived IDs and seed records;
    ``inputs`` carries references (ID + hash) to upstream artifacts.
    """
    if destination.exists():
        if is_complete(destination):
            raise ArtifactError(f"artifact already exists and is complete: {destination}")
        raise ArtifactError(
            f"destination exists but is not a complete artifact "
            f"(aborted attempt? remove it explicitly): {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        populate(staging)
        manifest: dict[str, Any] = {
            "schema": 1,
            "kind": kind,
            "identities": identities,
            "inputs": inputs,
            "files": _payload_ledger(staging),
        }
        if extra_manifest:
            overlap = set(extra_manifest) & set(manifest)
            if overlap:
                raise ArtifactError(f"extra manifest keys collide with envelope keys: {overlap}")
            manifest.update(extra_manifest)
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_checksums(staging)
        (staging / COMPLETE_NAME).touch()
        _fsync_tree(staging)
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _fsync_tree(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())


def read_manifest(directory: Path) -> dict[str, Any]:
    if not is_complete(directory):
        raise ArtifactError(f"artifact is not complete: {directory}")
    raw = (directory / MANIFEST_NAME).read_text(encoding="utf-8")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ArtifactError(f"malformed manifest in {directory}")
    return manifest


def verify_artifact(directory: Path) -> dict[str, Any]:
    """Recompute every checksum and return the manifest, or raise."""
    manifest = read_manifest(directory)
    recorded: dict[str, str] = {}
    for line in (directory / CHECKSUMS_NAME).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition("  ")
        if not digest or not relative or relative.startswith(("/", "..")):
            raise ArtifactError(f"malformed checksum line in {directory}: {line!r}")
        recorded[relative] = digest
    present = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and path.relative_to(directory).as_posix() not in {CHECKSUMS_NAME, COMPLETE_NAME}
    }
    missing = set(recorded) - present
    unlisted = present - set(recorded)
    if missing or unlisted:
        raise ArtifactError(
            f"checksum coverage mismatch in {directory}: missing={sorted(missing)} "
            f"unlisted={sorted(unlisted)}"
        )
    for relative, digest in recorded.items():
        actual = file_sha256(directory / relative)
        if actual != digest:
            raise ArtifactError(f"checksum mismatch for {relative} in {directory}")
    return manifest
