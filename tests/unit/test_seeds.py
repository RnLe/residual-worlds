"""Seed-tree determinism, disjointness, and pairing by namespace design."""

import numpy as np
import pytest
import torch

from residual_worlds.seeds import numpy_generator, seed_record, torch_generator

ROOT = 730241


def test_same_namespace_reproduces_stream() -> None:
    a = numpy_generator(ROOT, "data", "composite_standard", 0).standard_normal(8)
    b = numpy_generator(ROOT, "data", "composite_standard", 0).standard_normal(8)
    np.testing.assert_array_equal(a, b)


def test_different_namespaces_are_disjoint() -> None:
    a = numpy_generator(ROOT, "data", "composite_standard", 0).standard_normal(8)
    b = numpy_generator(ROOT, "data", "composite_standard", 1).standard_normal(8)
    c = numpy_generator(ROOT, "split", "composite_standard", 0).standard_normal(8)
    assert not np.allclose(a, b)
    assert not np.allclose(a, c)


def test_string_and_int_tokens_are_distinct() -> None:
    # A replicate index of 0 and the string "0" are different coordinates.
    assert seed_record(ROOT, "x", 0).digest_sha256 != seed_record(ROOT, "x", "0").digest_sha256


def test_token_concatenation_cannot_collide() -> None:
    # ("ab", "c") and ("a", "bc") must not produce the same stream.
    assert (
        seed_record(ROOT, "ab", "c").digest_sha256
        != seed_record(ROOT, "a", "bc").digest_sha256
    )


def test_root_seed_changes_everything() -> None:
    assert seed_record(1, "x").digest_sha256 != seed_record(2, "x").digest_sha256


def test_torch_generator_determinism() -> None:
    a = torch.randn(4, generator=torch_generator(ROOT, "model_init", "residual", 0, 0))
    b = torch.randn(4, generator=torch_generator(ROOT, "model_init", "residual", 0, 0))
    assert torch.equal(a, b)


def test_invalid_tokens_rejected() -> None:
    with pytest.raises(TypeError):
        seed_record(ROOT, 1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        seed_record(ROOT, True)  # type: ignore[arg-type]


def test_record_is_auditable() -> None:
    record = seed_record(ROOT, "cem_call", "paired_methods", "proto", "w", 2048, 0, "s", 17)
    assert record.tokens == ("cem_call", "paired_methods", "proto", "w", 2048, 0, "s", 17)
    assert record.numpy_seed >= 0
    assert 0 <= record.torch_seed < 2**63
