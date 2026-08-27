"""Canonical serialization and content-derived identifiers.

Every identity-bearing object in this project (dataset specs, scenario
content, analysis inputs, public bundles) is hashed the same way: the
payload is serialized with the RFC 8785 JSON Canonicalization Scheme and
digested with SHA-256 under a short domain tag. An identifier therefore
never depends on key order, whitespace, or platform float formatting,
and identifiers from different domains cannot collide by construction.

Constraints enforced here rather than trusted implicitly:

* string keys and values are normalized to Unicode NFC before hashing,
  and a key collision created by normalization is an error;
* non-finite floats are rejected (they have no canonical JSON form);
* integers outside the IEEE-754 interoperable range +/- (2^53 - 1) are
  rejected so that a JSON reimplementation cannot silently disagree.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from pathlib import Path
from typing import Any

import rfc8785

_ID_PREFIX = b"residual-worlds-id\x00"
_MAX_SAFE_INT = 2**53 - 1

# Domains are a closed set so a typo cannot mint a new identifier space.
KNOWN_DOMAINS = frozenset(
    {
        "protocol",
        "baseline",
        "spec",
        "scenario",
        "dataset",
        "prediction_set",
        "run",
        "condition",
        "evaluation_job",
        "prediction_job",
        "source_set",
        "analysis",
        "core_result",
        "report",
        "media",
        "public_trial",
        "public_bundle",
        "verification",
    }
)


def _normalize(value: Any) -> Any:
    """Recursively validate and NFC-normalize a JSON-compatible payload."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INT:
            raise ValueError(f"integer {value} exceeds the interoperable JSON range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats have no canonical JSON form")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"object keys must be strings, got {type(key).__name__}")
            nkey = unicodedata.normalize("NFC", key)
            if nkey in normalized:
                raise ValueError(f"duplicate key after NFC normalization: {nkey!r}")
            normalized[nkey] = _normalize(item)
        return normalized
    raise TypeError(f"value of type {type(value).__name__} is not JSON-serializable")


def canonical_json(payload: Any) -> bytes:
    """Serialize ``payload`` as RFC 8785 canonical JSON (UTF-8 bytes)."""
    return bytes(rfc8785.dumps(_normalize(payload)))


def content_id(domain: str, payload: Any) -> str:
    """Return the hex identifier of ``payload`` under a fixed domain tag."""
    if domain not in KNOWN_DOMAINS:
        raise ValueError(f"unknown identifier domain: {domain!r}")
    body = canonical_json({"domain": domain, "payload": payload})
    return hashlib.sha256(_ID_PREFIX + body).hexdigest()


def file_sha256(path: Path) -> str:
    """Plain SHA-256 of a file's bytes (distinct from a content identifier)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
