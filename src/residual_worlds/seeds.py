"""Deterministic seed derivation from named experimental coordinates.

No scientific function in this project draws from a global random state.
Every stream is derived from one root seed and an explicit namespace --
for example ``("data", "composite_standard", 0)`` for target-world data
generation of pipeline replicate 0. The namespace is serialized with the
same canonical JSON used for content identifiers, so a seed can be
reproduced from its recorded tokens alone and unrelated namespaces are
disjoint with cryptographic confidence.

Two consumers exist: NumPy ``Generator(PCG64DXSM)`` streams (data
generation, scenario sampling, CEM base noise, bootstrap resampling) and
``torch.Generator`` streams (network initialization, minibatch
sampling). Both integers come from one SHA-256 digest of the namespace,
so recording the token list fully determines both.

Pairing between the residual and black-box methods is achieved purely
through namespace design: paired streams (bootstrap units, minibatch
order, CEM noise) simply do not contain a method token.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from residual_worlds.identity import canonical_json

_SEED_PREFIX = b"residual-worlds-seed\x00"

Token = str | int


@dataclass(frozen=True)
class SeedRecord:
    """Everything needed to reproduce and audit one derived stream."""

    root_seed: int
    tokens: tuple[Token, ...]
    digest_sha256: str
    numpy_seed: int
    torch_seed: int


def _digest(root_seed: int, tokens: tuple[Token, ...]) -> bytes:
    if root_seed < 0:
        raise ValueError("root seed must be non-negative")
    typed: list[dict[str, Token]] = []
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, (str, int)):
            raise TypeError(f"seed tokens must be str or int, got {type(token).__name__}")
        typed.append({"s": token} if isinstance(token, str) else {"i": token})
    body = canonical_json({"schema": 1, "root_seed": root_seed, "tokens": typed})
    return hashlib.sha256(_SEED_PREFIX + body).digest()


def seed_record(root_seed: int, *tokens: Token) -> SeedRecord:
    """Derive the seed pair for a namespace and return its audit record."""
    digest = _digest(root_seed, tokens)
    numpy_seed = int.from_bytes(digest[:16], "big")
    torch_seed = int.from_bytes(digest[16:24], "big") & (2**63 - 1)
    return SeedRecord(
        root_seed=root_seed,
        tokens=tokens,
        digest_sha256=digest.hex(),
        numpy_seed=numpy_seed,
        torch_seed=torch_seed,
    )


def numpy_generator(root_seed: int, *tokens: Token) -> np.random.Generator:
    """Fresh ``PCG64DXSM`` generator for the given namespace."""
    record = seed_record(root_seed, *tokens)
    return np.random.Generator(np.random.PCG64DXSM(record.numpy_seed))


def torch_generator(root_seed: int, *tokens: Token) -> torch.Generator:
    """Fresh CPU ``torch.Generator`` for the given namespace."""
    record = seed_record(root_seed, *tokens)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(record.torch_seed)
    return generator
