"""The committed golden numbers must be what the physics produces today."""

import json
from pathlib import Path

from residual_worlds.config import load_contract
from residual_worlds.media.golden import build_arm_golden
from residual_worlds.media.loop import simulate_loop
from residual_worlds.paths import repository_root

ROOT = repository_root()
FIXTURE = ROOT / "tests" / "fixtures" / "arm_golden.json"


def test_committed_golden_matches_regeneration() -> None:
    contract = load_contract(ROOT / "configs" / "experiment_contract.yaml")
    fresh = json.loads(json.dumps(build_arm_golden(contract)))
    committed = json.loads(FIXTURE.read_text())
    assert fresh == committed


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
