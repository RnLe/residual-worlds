"""The committed golden numbers must be what the physics produces today."""

import json
import math
from pathlib import Path
from typing import Any

from residual_worlds.config import load_contract
from residual_worlds.media.golden import build_arm_golden
from residual_worlds.media.loop import simulate_loop
from residual_worlds.paths import repository_root

ROOT = repository_root()
FIXTURE = ROOT / "tests" / "fixtures" / "arm_golden.json"

# Two machines order the same floating-point work differently, so the last
# bits of a 140-step rollout do not travel. These tolerances sit far below
# any change worth noticing and far above that noise.
REL_TOL = 1e-9
ABS_TOL = 1e-12


def assert_matches(fresh: Any, committed: Any, path: str = "root") -> None:
    """Compare two decoded payloads, numbers by tolerance and the rest exactly."""
    if isinstance(fresh, dict):
        assert isinstance(committed, dict), path
        assert fresh.keys() == committed.keys(), path
        for key in fresh:
            assert_matches(fresh[key], committed[key], f"{path}.{key}")
    elif isinstance(fresh, list):
        assert isinstance(committed, list), path
        assert len(fresh) == len(committed), path
        for index, (left, right) in enumerate(zip(fresh, committed, strict=True)):
            assert_matches(left, right, f"{path}[{index}]")
    elif isinstance(fresh, float) or isinstance(committed, float):
        assert math.isclose(fresh, committed, rel_tol=REL_TOL, abs_tol=ABS_TOL), (
            f"{path}: {fresh!r} against {committed!r}"
        )
    else:
        assert fresh == committed, path


def test_committed_golden_matches_regeneration() -> None:
    contract = load_contract(ROOT / "configs" / "experiment_contract.yaml")
    fresh = json.loads(json.dumps(build_arm_golden(contract)))
    committed = json.loads(FIXTURE.read_text())
    assert_matches(fresh, committed)


def test_loop_is_deterministic_and_inside_limits(tmp_path: Path) -> None:
    contract = load_contract(ROOT / "configs" / "experiment_contract.yaml")
    a = simulate_loop(contract)
    b = simulate_loop(contract)
    assert (a.states == b.states).all()
    arm = contract.arm
    for j in range(2):
        assert a.states[:, j].min() >= arm.q_min_rad[j]
        assert a.states[:, j].max() <= arm.q_max_rad[j]
        assert abs(a.states[:, 2 + j]).max() < arm.speed_limit_rad_s[j]
        assert abs(a.actions[:, j]).max() <= arm.torque_limit_nm[j]
