"""Task automaton and true-environment event precedence."""

import math

import numpy as np
import pytest
import torch

from residual_worlds.config import load_contract
from residual_worlds.paths import repository_root
from residual_worlds.physics.nominal import gravity_vector, state_acceleration
from residual_worlds.task.reaching import TaskRules, TrueArmEnv, advance_task
from residual_worlds.types import Scenario

pytestmark = pytest.mark.scientific

CONTRACT = load_contract(repository_root() / "configs" / "experiment_contract.yaml")
ARM = CONTRACT.arm
RULES = TaskRules.from_config(CONTRACT.task)


def _nominal_accel(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return state_acceleration(state, action, ARM)


def _ee(q1: float, q2: float) -> tuple[float, float]:
    from residual_worlds.physics.kinematics import end_effector_position

    p = end_effector_position(torch.tensor([q1, q2], dtype=torch.float64), ARM)
    return float(p[0]), float(p[1])


def _scenario(
    initial=(0.8, -0.4, 0.0, 0.0),
    targets=None,
    obstacle=(0.0, -0.9, 0.05),  # far below the workspace: irrelevant
    timeout=160,
) -> Scenario:
    if targets is None:
        targets = (_ee(1.0, -0.5), _ee(1.3, -0.3), _ee(0.6, -0.7))
    return Scenario(
        scenario_id="test",
        bank="test",
        index=0,
        stratum_id=0,
        target_order=(0, 1, 2),
        obstacle_chord_index=0,
        initial_state=initial,
        targets_xy_m=targets,
        obstacle_xy_radius_m=obstacle,
        timeout_steps=timeout,
    )


def _resting_env(scenario: Scenario) -> TrueArmEnv:
    return TrueArmEnv(
        _nominal_accel,
        scenario,
        RULES,
        ARM,
        CONTRACT.numerics.control_dt_s,
        substeps=2,
    )


# ---------------------------------------------------------------------------
# Pure automaton behavior


def test_dwell_accumulates_and_completes() -> None:
    target = _ee(0.8, -0.4)
    targets = torch.tensor([[target[0], target[1]], [0.9, 0.9], [-0.9, 0.9]]).to(torch.float64)
    q = torch.tensor([0.8, -0.4], dtype=torch.float64)
    qd = torch.zeros(2, dtype=torch.float64)
    index = torch.tensor(0)
    dwell = torch.tensor(0)
    for expected in (1, 2, 3):
        index, dwell = advance_task(q, qd, index, dwell, targets, RULES, ARM)
        assert (int(index), int(dwell)) == (0, expected)
    index, dwell = advance_task(q, qd, index, dwell, targets, RULES, ARM)
    assert (int(index), int(dwell)) == (1, 0)  # fourth consecutive step completes


def test_dwell_resets_when_leaving_or_moving_fast() -> None:
    target = _ee(0.8, -0.4)
    targets = torch.tensor([[target[0], target[1]], [0.9, 0.9], [-0.9, 0.9]]).to(torch.float64)
    q = torch.tensor([0.8, -0.4], dtype=torch.float64)
    slow = torch.zeros(2, dtype=torch.float64)
    fast = torch.tensor([2.0, 0.0], dtype=torch.float64)
    index, dwell = advance_task(q, slow, torch.tensor(0), torch.tensor(2), targets, RULES, ARM)
    assert (int(index), int(dwell)) == (0, 3)
    index, dwell = advance_task(q, fast, torch.tensor(0), torch.tensor(3), targets, RULES, ARM)
    assert (int(index), int(dwell)) == (0, 0)  # speed threshold violated: reset


def test_completed_state_is_absorbing() -> None:
    targets = torch.zeros(3, 2, dtype=torch.float64)
    q = torch.tensor([0.8, -0.4], dtype=torch.float64)
    qd = torch.zeros(2, dtype=torch.float64)
    index, dwell = advance_task(
        q, qd, torch.tensor(3), torch.tensor(0), targets, RULES, ARM
    )
    assert (int(index), int(dwell)) == (3, 0)


def test_batched_automaton_advances_independently() -> None:
    target = _ee(0.8, -0.4)
    targets = torch.tensor([[target[0], target[1]], [0.9, 0.9], [-0.9, 0.9]]).to(torch.float64)
    q = torch.stack(
        [torch.tensor([0.8, -0.4]), torch.tensor([0.1, 0.1])], dim=0
    ).to(torch.float64)
    qd = torch.zeros(2, 2, dtype=torch.float64)
    index = torch.zeros(2, dtype=torch.long)
    dwell = torch.tensor([3, 3])
    new_index, new_dwell = advance_task(
        q, qd, index, dwell, targets.expand(2, 3, 2), RULES, ARM
    )
    assert new_index.tolist() == [1, 0]  # only the first is at its target
    assert new_dwell.tolist() == [0, 0]


# ---------------------------------------------------------------------------
# Environment event precedence


def test_gravity_hold_reaches_success() -> None:
    # Place all three targets at the initial pose and hold the arm there
    # with exact gravity compensation: dwell must accumulate to success.
    initial_pose = (1.4, -0.3)
    scenario = _scenario(
        initial=(initial_pose[0], initial_pose[1], 0.0, 0.0),
        targets=(_ee(*initial_pose), _ee(*initial_pose), _ee(*initial_pose)),
    )
    env = _resting_env(scenario)
    env.reset()
    hold = gravity_vector(torch.tensor(initial_pose, dtype=torch.float64), ARM).numpy()
    terminated = truncated = False
    steps = 0
    reason = ""
    while not (terminated or truncated):
        _obs, _reward, terminated, truncated, info = env.step(hold)
        reason = info.get("reason", "")
        steps += 1
        assert steps < 40
    assert terminated and not truncated
    assert reason == "SUCCESS"
    # 3 targets x 4 dwell steps = 12 executed transitions.
    assert steps == 3 * RULES.target_dwell_steps


def test_timeout_is_truncation_with_zero_targets() -> None:
    scenario = _scenario(initial=(1.4, -0.3, 0.0, 0.0), timeout=10)
    env = _resting_env(scenario)
    env.reset()
    hold = gravity_vector(
        torch.tensor(scenario.initial_state[:2], dtype=torch.float64), ARM
    ).numpy()
    for _ in range(9):
        _obs, _reward, terminated, truncated, info = env.step(hold)
        assert not terminated and not truncated
    _obs, _reward, terminated, truncated, info = env.step(hold)
    assert truncated and not terminated
    assert info["reason"] == "TIMEOUT_ZERO_TARGETS"


def test_success_allowed_on_final_step() -> None:
    # Timeout equal to exactly the required dwell steps: the completing
    # transition into the last allowed step must be SUCCESS, not timeout.
    initial_pose = (1.4, -0.3)
    scenario = _scenario(
        initial=(initial_pose[0], initial_pose[1], 0.0, 0.0),
        targets=(_ee(*initial_pose), _ee(*initial_pose), _ee(*initial_pose)),
        timeout=3 * RULES.target_dwell_steps,
    )
    env = _resting_env(scenario)
    env.reset()
    hold = gravity_vector(torch.tensor(initial_pose, dtype=torch.float64), ARM).numpy()
    outcome = None
    for _ in range(scenario.timeout_steps):
        _obs, _reward, terminated, truncated, info = env.step(hold)
        if terminated or truncated:
            outcome = (terminated, truncated, info["reason"])
            break
    assert outcome == (True, False, "SUCCESS")


def test_collision_terminates_with_reason() -> None:
    # Obstacle directly on the initial arm: first transition collides.
    initial_pose = (1.4, -0.3)
    tip = _ee(*initial_pose)
    scenario = _scenario(
        initial=(initial_pose[0], initial_pose[1], 0.0, 0.0),
        obstacle=(tip[0], tip[1], 0.06),
    )
    env = _resting_env(scenario)
    env.reset()
    _obs, _reward, terminated, truncated, info = env.step(np.zeros(2))
    assert terminated and not truncated
    assert info["reason"] == "OBSTACLE_COLLISION"


def test_speed_limit_terminates_with_reason() -> None:
    # Shoulder already fast in the negative direction; gravity and full
    # negative torque both accelerate it past the 8 rad/s limit.
    scenario = _scenario(initial=(0.9, 0.0, -7.7, 0.0))
    env = _resting_env(scenario)
    env.reset()
    _obs, _reward, terminated, truncated, info = env.step(np.array([-4.0, 0.0]))
    assert terminated
    assert info["reason"] == "HARD_LIMIT_OR_SPEED"


def test_env_is_deterministic() -> None:
    scenario = _scenario()
    trace_a, trace_b = [], []
    for trace in (trace_a, trace_b):
        env = _resting_env(scenario)
        env.reset()
        for step in range(20):
            action = np.array([math.sin(step * 0.3), -math.cos(step * 0.2)])
            obs, _r, terminated, truncated, _info = env.step(action)
            trace.append(obs.copy())
            if terminated or truncated:
                break
    np.testing.assert_array_equal(np.array(trace_a), np.array(trace_b))
