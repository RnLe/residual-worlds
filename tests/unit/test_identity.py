"""Canonical JSON and content-identifier behavior."""

import math

import pytest

from residual_worlds.identity import canonical_json, content_id, file_sha256


def test_key_order_does_not_change_identity() -> None:
    a = {"x": 1, "y": [1, 2, 3], "z": {"a": True, "b": None}}
    b = {"z": {"b": None, "a": True}, "y": [1, 2, 3], "x": 1}
    assert content_id("spec", a) == content_id("spec", b)


def test_value_change_changes_identity() -> None:
    assert content_id("spec", {"x": 1}) != content_id("spec", {"x": 2})


def test_domain_separates_identifiers() -> None:
    payload = {"x": 1}
    assert content_id("spec", payload) != content_id("scenario", payload)


def test_unknown_domain_rejected() -> None:
    with pytest.raises(ValueError):
        content_id("not-a-domain", {})


def test_nonfinite_floats_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})
    with pytest.raises(ValueError):
        canonical_json({"x": math.inf})


def test_huge_integers_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": 2**53})
    # The boundary itself is representable.
    canonical_json({"x": 2**53 - 1})


def test_nfc_normalization_is_applied() -> None:
    # "é" precomposed vs combining-accent form must hash identically.
    assert content_id("spec", {"k": "é"}) == content_id("spec", {"k": "é"})


def test_nfc_key_collision_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_json({"é": 1, "é": 2})


def test_number_formatting_is_canonical() -> None:
    # RFC 8785 requires shortest round-trip float serialization.
    assert canonical_json({"x": 1.0}) == b'{"x":1}'
    assert canonical_json({"x": 0.1}) == b'{"x":0.1}'


def test_file_sha256(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"residual worlds")
    assert len(file_sha256(path)) == 64
    path2 = tmp_path / "payload2.bin"
    path2.write_bytes(b"residual worlds")
    assert file_sha256(path) == file_sha256(path2)
