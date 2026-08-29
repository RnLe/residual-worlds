"""Verification of a public result bundle before any surface consumes it.

Fails closed: unknown status values, missing required payloads, or
checksum mismatches reject the bundle rather than best-effort
rendering. Production surfaces additionally require
``content_status: final`` -- the watermarked schematic fixture can
never masquerade as evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from residual_worlds.provenance import verify_artifact
from residual_worlds.release.schema import (
    CONTENT_STATUSES,
    INTERPRETATION_STATES,
    KEY_RESULTS_REQUIRED,
    PRIMARY_REQUIRED,
    REQUIRED_FILES,
)


class BundleError(RuntimeError):
    pass


def verify_bundle(
    bundle_directory: Path, require_content_status: str | None = None
) -> dict[str, Any]:
    """Validate structure, checksums, schema, and status enums."""
    manifest = verify_artifact(bundle_directory)

    missing = [
        relative
        for relative in REQUIRED_FILES
        if not (bundle_directory / relative).exists()
    ]
    if missing:
        raise BundleError(f"bundle is missing required payloads: {missing}")

    key_results = json.loads(
        (bundle_directory / "summary/key_results.json").read_text(encoding="utf-8")
    )
    for key in KEY_RESULTS_REQUIRED:
        if key not in key_results:
            raise BundleError(f"key_results.json lacks required field {key!r}")
    for key in PRIMARY_REQUIRED:
        if key not in key_results["primary"]:
            raise BundleError(f"key_results primary block lacks field {key!r}")

    content_status = key_results["content_status"]
    if content_status not in CONTENT_STATUSES:
        raise BundleError(f"unknown content_status {content_status!r}")
    interpretation_state = key_results["interpretation_state"]
    if interpretation_state not in INTERPRETATION_STATES:
        raise BundleError(f"unknown interpretation_state {interpretation_state!r}")
    if manifest["identities"].get("content_status") != content_status:
        raise BundleError("manifest and key_results disagree on content_status")

    if require_content_status is not None and content_status != require_content_status:
        raise BundleError(
            f"bundle content_status is {content_status!r}; "
            f"{require_content_status!r} is required"
        )

    registry = json.loads(
        (bundle_directory / "figures/figure_registry.json").read_text(encoding="utf-8")
    )
    for entry in registry:
        for name in entry["files"]:
            if not (bundle_directory / "figures" / name).exists():
                raise BundleError(f"figure registry references missing file {name}")

    return {
        "content_status": content_status,
        "interpretation_state": interpretation_state,
        "analysis_id": key_results["analysis_id"],
        "core_id": manifest["identities"].get("core_id"),
        "figures": len(registry),
    }
