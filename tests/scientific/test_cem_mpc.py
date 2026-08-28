"""CEM semantics, warm start, paired noise, and closed-loop competence."""

import pytest
import torch

from residual_worlds.config import load_contract
from residual_worlds.models.base import Stepper
from residual_worlds.paths import repository_root
from residual_worlds.physics.nominal import state_acceleration
from residual_worlds.planning.cem import (
    CEMSettings,
    cem_optimize,
    expand_knots,
    latent_to_action,
    rank_candidates,
)
from residual_worlds.planning.costs import ScenarioTensors
from residual_worlds.planning.mpc import (
    ControllerSettings,
    MPCController,
    cem_base_noise,
    make_noise_fn,
)
from residual_worlds.task.reaching import TaskRules, TrueArmEnv
from residual_worlds.types import Scenario, TaskState

pytestmark = pytest.mark.scientific

SMOKE = load_contract(repository_root() / "configs" / "smoke.yaml")
ARM = SMOKE.arm
ROOT = SMOKE.numerics.root_seed


def test_latent_transform_respects_torque_limits() -> None:
    limit = torch.tensor(ARM.torque_limit_nm, dtype=torch.float32)
    latent = torch.randn(1000, 2) * 50.0
    actions = latent_to_action(latent, limit)
    assert torch.all(torch.abs(actions) <= limit + 1e-6)


def test_expand_knots_repeats_pairs() -> None:
    knots = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    expanded = expand_knots(knots, 10)
    assert expanded.shape == (10, 2)
    torch.testing.assert_close(expanded[0], expanded[1])
    torch.testing.assert_close(expanded[0], knots[0])
    torch.testing.assert_close(expanded[8], knots[4])


def test_rank_candidates_order() -> None:
    costs = torch.tensor([3.0, 1.0, float("nan"), 2.0, 1.0], dtype=torch.float32)
    invalid = torch.tensor([False, False, False, True, True])
    order = rank_candidates(costs, invalid)
    # Valid by cost (ties by index), then invalid by cost, NaN last.
    assert order.tolist() == [1, 0, 2, 4, 3]


def test_cem_solves_quadratic_toy() -> None:
    # Minimize sum (z - z*)^2 over latent knots: a pure optimizer check.
    target = torch.tensor([[0.7, -0.3]] * 5, dtype=torch.float32)

    def scorer(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        costs = torch.sum((latents - target) ** 2, dim=(-2, -1))
        return costs, torch.zeros(latents.shape[0], dtype=torch.bool)

    settings = CEMSettings(
        candidates=64,
        elites=8,
        iterations=8,
        action_knots=5,
        initial_latent_std=1.0,
        latent_std_floor=0.05,
        old_distribution_retention=0.2,
    )
    noise, _digest = cem_base_noise(ROOT, ("test_toy",), (8, 64, 5, 2))
    result = cem_optimize(scorer, torch.zeros(5, 2), noise, settings)
    assert float(torch.max(torch.abs(result.final_mean - target))) < 0.12
    # Best cost is non-increasing-ish and small by the end.
    assert float(result.best_cost_by_iteration[-1]) < 0.05


def test_cem_noise_is_deterministic_and_hashed() -> None:
    a, digest_a = cem_base_noise(ROOT, ("cem_call", "x", 3), (2, 4, 5, 2))
    b, digest_b = cem_base_noise(ROOT, ("cem_call", "x", 3), (2, 4, 5, 2))
    torch.testing.assert_close(a, b)
    assert digest_a == digest_b
    _c, digest_c = cem_base_noise(ROOT, ("cem_call", "x", 4), (2, 4, 5, 2))
    assert digest_c != digest_a


def _scenario_at_pose() -> Scenario:
    # A high-workspace pose with ample gravity authority; see
    # docs/protocol_notes.md for why low poses are marginal under the
    # draft task parameters (a calibration-gate matter, not a planner
    # property).
    from residual_worlds.physics.kinematics import end_effector_position

    def ee(q1: float, q2: float) -> tuple[float, float]:
        p = end_effector_position(torch.tensor([q1, q2], dtype=torch.float64), ARM)
        return float(p[0]), float(p[1])

    target = ee(1.7, -0.4)
    return Scenario(
        scenario_id="mpc-test",
        bank="test",
        index=0,
        stratum_id=0,
        target_order=(0, 1, 2),
        obstacle_chord_index=0,
        initial_state=(1.8, -0.5, 0.0, 0.0),
        targets_xy_m=(target, target, target),
        obstacle_xy_radius_m=(0.0, -0.9, 0.05),  # far outside the workspace
        timeout_steps=140,
    )


def _controller(noise_namespace: tuple[str | int, ...]) -> MPCController:
    # A mid-strength profile: strong enough for closed-loop competence
    # on the easy test task, small enough for CPU test time.
    cem = CEMSettings(
        candidates=128,
        elites=16,
        iterations=4,
        action_knots=SMOKE.planning.action_knots,
        initial_latent_std=SMOKE.planning.initial_latent_std,
        latent_std_floor=SMOKE.planning.latent_std_floor,
        old_distribution_retention=SMOKE.planning.old_distribution_retention,
    )
    settings = ControllerSettings(
        horizon_steps=SMOKE.planning.horizon_steps,
        action_knots=SMOKE.planning.action_knots,
        execute_actions_per_plan=1,
        replan_every_steps=1,
        cem=cem,
    )
    shape = (cem.iterations, cem.candidates, cem.action_knots, 2)
    scenario = _scenario_at_pose()
    return MPCController(
        members=[lambda s, a: state_acceleration(s, a, ARM)],
        scenario_tensors=ScenarioTensors.from_scenario(scenario, torch.float32),
        weights=SMOKE.task.cost,
        rules=TaskRules.from_config(SMOKE.task),
        arm=ARM,
        stepper=Stepper.from_contract(SMOKE),
        settings=settings,
        noise_fn=make_noise_fn(ROOT, noise_namespace, shape),
    )


def test_plan_is_deterministic() -> None:
    state = torch.tensor([1.4, -0.3, 0.0, 0.0], dtype=torch.float64)
    task = TaskState(target_index=0, dwell_count=0, previous_action=(0.0, 0.0))
    plan_a = _controller(("cem_call", "det")).plan(state, task, 0)
    plan_b = _controller(("cem_call", "det")).plan(state, task, 0)
    torch.testing.assert_close(plan_a.actions, plan_b.actions)
    assert plan_a.predicted_cost == plan_b.predicted_cost
    assert plan_a.diagnostics["noise_sha256"] == plan_b.diagnostics["noise_sha256"]


def test_paired_methods_share_primitive_noise() -> None:
    # Two different "methods" (different member dynamics) with the same
    # method-free namespace receive identical primitive noise hashes.
    state = torch.tensor([1.4, -0.3, 0.0, 0.0], dtype=torch.float64)
    task = TaskState(target_index=0, dwell_count=0, previous_action=(0.0, 0.0))
    plan_a = _controller(("cem_call", "paired")).plan(state, task, 5)

    perturbed = _controller(("cem_call", "paired"))
    perturbed._members = (  # type: ignore[attr-defined]
        lambda s, a: state_acceleration(s, a, ARM) * 1.1,
    )
    plan_b = perturbed.plan(state, task, 5)
    assert plan_a.diagnostics["noise_sha256"] == plan_b.diagnostics["noise_sha256"]
    # Their chosen actions may of course differ.


def test_warm_start_shift_math() -> None:
    controller = _controller(("cem_call", "warm"))
    mean = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    controller._latent_mean = mean  # type: ignore[attr-defined]
    shifted = controller._warm_start_shift()  # type: ignore[attr-defined]
    # Expand: [k0 k0 k1 k1 k2 k2 k3 k3 k4 k4]; shift by 1:
    # [k0 k1 k1 k2 k2 k3 k3 k4 k4 k4]; pairwise averages:
    # [(k0+k1)/2, (k1+k2)/2, (k2+k3)/2, (k3+k4)/2, k4].
    expected = torch.stack(
        [
            (mean[0] + mean[1]) / 2,
            (mean[1] + mean[2]) / 2,
            (mean[2] + mean[3]) / 2,
            (mean[3] + mean[4]) / 2,
            mean[4],
        ]
    )
    torch.testing.assert_close(shifted, expected)


@pytest.mark.slow
def test_exact_model_mpc_completes_task() -> None:
    # The controller with the exact (nominal-world) model must complete
    # the three-dwell task on an obstacle-free scenario: closed-loop
    # competence of the whole planner stack.
    scenario = _scenario_at_pose()
    controller = _controller(("cem_call", "closedloop"))
    env = TrueArmEnv(
        lambda s, a: state_acceleration(s, a, ARM),
        scenario,
        TaskRules.from_config(SMOKE.task),
        ARM,
        SMOKE.numerics.control_dt_s,
        substeps=SMOKE.numerics.substeps_per_control_step,
    )
    env.reset()
    controller.reset()
    previous_action = (0.0, 0.0)
    reason = ""
    for step in range(scenario.timeout_steps):
        state = torch.from_numpy(env.state)
        task = TaskState(
            target_index=env.target_index,
            dwell_count=env.dwell_count,
            previous_action=previous_action,
        )
        plan = controller.plan(state, task, step)
        action = plan.actions[0].to(torch.float64).numpy()
        _obs, _r, terminated, truncated, info = env.step(action)
        previous_action = (float(action[0]), float(action[1]))
        if terminated or truncated:
            reason = info["reason"]
            break
    assert reason == "SUCCESS", f"episode ended with {reason!r} at step {step}"
