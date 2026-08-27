"""Simulator verification: analytic identities, energy behavior, convergence.

A rendered arm that "looks right" proves nothing. This suite checks the
nominal implementation against independent evidence:

* structural identities (mass-matrix symmetry and positive definiteness,
  gravity as the exact gradient of the potential, solve residuals,
  batched-vs-scalar-reference agreement);
* physical behavior (energy conservation without dissipation, monotone
  dissipation with damping, equilibrium under exactly balancing torque);
* numerical accuracy (RK4 one-step and eight-second errors against a
  tight adaptive ``solve_ivp`` reference, fourth-order convergence,
  float32 planning-wrapper deviation from float64 truth).

Results are written as an immutable artifact with the energy and
convergence plots, and summarized against the integration gate in the
contract. Tolerances come from the contract, not from whatever the
implementation happens to achieve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.integrate import solve_ivp

from residual_worlds.config import ExperimentContract
from residual_worlds.identity import content_id
from residual_worlds.paths import verification_dir
from residual_worlds.physics import nominal
from residual_worlds.physics.integrators import rk4_transition
from residual_worlds.provenance import is_complete, verify_artifact, write_artifact
from residual_worlds.seeds import numpy_generator
from residual_worlds.types import ArmParameters

SUITE_VERSION = 2


def normalized_state_error(delta: torch.Tensor | np.ndarray) -> np.ndarray:
    """Scale-normalized state error: angles / 1 rad, velocities / 4 rad/s."""
    array = np.asarray(delta, dtype=np.float64)
    dq = array[..., :2] / 1.0
    dqd = array[..., 2:] / 4.0
    return np.sqrt(0.25 * (np.sum(dq**2, axis=-1) + np.sum(dqd**2, axis=-1)))


def _sample_states(
    contract: ExperimentContract, count: int, speed_scale: float = 0.8
) -> torch.Tensor:
    """Random valid states inside the hard limits (float64)."""
    rng = numpy_generator(contract.numerics.root_seed, "verification", "states", SUITE_VERSION)
    arm = contract.arm
    q = np.stack(
        [
            rng.uniform(arm.q_min_rad[j] + 0.05, arm.q_max_rad[j] - 0.05, size=count)
            for j in range(2)
        ],
        axis=-1,
    )
    qd = np.stack(
        [
            rng.uniform(
                -speed_scale * arm.speed_limit_rad_s[j],
                speed_scale * arm.speed_limit_rad_s[j],
                size=count,
            )
            for j in range(2)
        ],
        axis=-1,
    )
    return torch.from_numpy(np.concatenate([q, qd], axis=-1))


def _sample_actions(contract: ExperimentContract, count: int) -> torch.Tensor:
    rng = numpy_generator(contract.numerics.root_seed, "verification", "actions", SUITE_VERSION)
    arm = contract.arm
    u = np.stack(
        [
            rng.uniform(-arm.torque_limit_nm[j], arm.torque_limit_nm[j], size=count)
            for j in range(2)
        ],
        axis=-1,
    )
    return torch.from_numpy(u)


def _solve_ivp_transition(
    state: np.ndarray,
    action: np.ndarray,
    duration: float,
    arm: ArmParameters,
    rtol: float,
    atol: float,
) -> np.ndarray:
    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        qdd = nominal.acceleration_reference_numpy(y[:2], y[2:], action, arm)
        return np.concatenate([y[2:], qdd])

    solution = solve_ivp(
        rhs, (0.0, duration), state, method="DOP853", rtol=rtol, atol=atol, dense_output=False
    )
    if not solution.success:
        raise RuntimeError(f"solve_ivp reference failed: {solution.message}")
    return np.asarray(solution.y[:, -1], dtype=np.float64)


def check_mass_matrix(contract: ExperimentContract, count: int = 256) -> dict[str, Any]:
    states = _sample_states(contract, count)
    m = nominal.mass_matrix(states[..., :2], contract.arm)
    asymmetry = float(torch.max(torch.abs(m - m.transpose(-1, -2))))
    eigenvalues = torch.linalg.eigvalsh(m)
    minimum_eigenvalue = float(torch.min(eigenvalues))
    rhs = _sample_actions(contract, count)
    qdd = torch.linalg.solve(m, rhs.unsqueeze(-1)).squeeze(-1)
    residual = float(torch.max(torch.abs(torch.einsum("...ij,...j->...i", m, qdd) - rhs)))
    return {
        "max_asymmetry": asymmetry,
        "min_eigenvalue": minimum_eigenvalue,
        "max_solve_residual": residual,
        "passed": asymmetry < 1e-12 and minimum_eigenvalue > 0.0 and residual < 1e-10,
    }


def check_batched_matches_reference(
    contract: ExperimentContract, count: int = 256
) -> dict[str, Any]:
    states = _sample_states(contract, count)
    actions = _sample_actions(contract, count)
    batched = nominal.state_acceleration(states, actions, contract.arm).numpy()
    worst = 0.0
    for index in range(count):
        reference = nominal.acceleration_reference_numpy(
            states[index, :2].numpy(), states[index, 2:].numpy(), actions[index].numpy(),
            contract.arm,
        )
        worst = max(worst, float(np.max(np.abs(batched[index] - reference))))
    return {"max_abs_difference": worst, "passed": worst < 1e-10}


def check_gravity_gradient(contract: ExperimentContract, count: int = 256) -> dict[str, Any]:
    states = _sample_states(contract, count)
    q = states[..., :2].clone().requires_grad_(True)
    potential = nominal.potential_energy(q, contract.arm).sum()
    (autograd_gradient,) = torch.autograd.grad(potential, q)
    analytic = nominal.gravity_vector(q.detach(), contract.arm)
    autograd_error = float(torch.max(torch.abs(autograd_gradient - analytic)))

    # Independent central finite differences.
    eps = 1e-6
    finite_error = 0.0
    q_detached = q.detach()
    for joint in range(2):
        offset = torch.zeros_like(q_detached)
        offset[..., joint] = eps
        numeric = (
            nominal.potential_energy(q_detached + offset, contract.arm)
            - nominal.potential_energy(q_detached - offset, contract.arm)
        ) / (2 * eps)
        finite_error = max(
            finite_error, float(torch.max(torch.abs(numeric - analytic[..., joint])))
        )
    return {
        "max_autograd_error": autograd_error,
        "max_finite_difference_error": finite_error,
        "passed": autograd_error < 1e-9 and finite_error < 1e-5,
    }


def check_energy_conservation(
    contract: ExperimentContract, duration_s: float = 2.0
) -> dict[str, Any]:
    """Conservative system (no damping, no input): T + V drift shrinks ~ h^4."""
    arm = contract.arm
    conservative = ArmParameters(
        link_lengths_m=arm.link_lengths_m,
        com_lengths_m=arm.com_lengths_m,
        masses_kg=arm.masses_kg,
        inertias_kg_m2=arm.inertias_kg_m2,
        viscous_nm_s_rad=(0.0, 0.0),
        gravity_m_s2=arm.gravity_m_s2,
        torque_limit_nm=arm.torque_limit_nm,
        q_min_rad=arm.q_min_rad,
        q_max_rad=arm.q_max_rad,
        speed_limit_rad_s=arm.speed_limit_rad_s,
    )
    state = torch.tensor([0.9, -0.6, 0.4, -0.3], dtype=torch.float64)
    action = torch.zeros(2, dtype=torch.float64)
    dt = contract.numerics.control_dt_s
    steps = int(round(duration_s / dt))

    def accel(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return nominal.state_acceleration(s, a, conservative)

    def total_energy(s: torch.Tensor) -> float:
        return float(
            nominal.kinetic_energy(s[:2], s[2:], conservative)
            + nominal.potential_energy(s[:2], conservative)
        )

    drifts: dict[int, float] = {}
    traces: dict[int, list[float]] = {}
    for substeps in (1, 2, 4, 8):
        current = state.clone()
        initial_energy = total_energy(current)
        trace = [initial_energy]
        worst = 0.0
        for _ in range(steps):
            current = rk4_transition(accel, current, action, dt, substeps)
            energy = total_energy(current)
            trace.append(energy)
            worst = max(worst, abs(energy - initial_energy))
        drifts[substeps] = worst
        traces[substeps] = trace

    # Fourth-order behavior: doubling the substep count should shrink the
    # drift by roughly 2^4; anything above 8x counts as passing here. The
    # absolute drift at the finest substep is RK4 truncation on a chaotic
    # free double pendulum, so it is bounded loosely (1e-3 J on ~10 J
    # energies); the convergence *rate* is the actual correctness check.
    ratios = [drifts[1] / drifts[2], drifts[2] / drifts[4], drifts[4] / drifts[8]]
    passed = all(ratio > 8.0 for ratio in ratios) and drifts[8] < 1e-3
    return {
        "drift_by_substeps": {str(k): v for k, v in drifts.items()},
        "convergence_ratios": ratios,
        "passed": passed,
        "_energy_traces": traces,
    }


def check_dissipation_monotonic(
    contract: ExperimentContract, duration_s: float = 4.0
) -> dict[str, Any]:
    """With viscous damping and zero input, T + V never increases."""
    arm = contract.arm
    state = torch.tensor([1.1, 0.8, 1.5, -2.0], dtype=torch.float64)
    action = torch.zeros(2, dtype=torch.float64)
    dt = contract.numerics.control_dt_s
    steps = int(round(duration_s / dt))

    def accel(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return nominal.state_acceleration(s, a, arm)

    def total_energy(s: torch.Tensor) -> float:
        return float(
            nominal.kinetic_energy(s[:2], s[2:], arm) + nominal.potential_energy(s[:2], arm)
        )

    current = state.clone()
    energies = [total_energy(current)]
    for _ in range(steps):
        current = rk4_transition(accel, current, action, dt, 8)
        energies.append(total_energy(current))
    increases = [b - a for a, b in zip(energies[:-1], energies[1:], strict=True) if b > a]
    max_increase = max(increases, default=0.0)
    return {
        "max_energy_increase": max_increase,
        "passed": max_increase < 1e-9,
        "_energy_trace": energies,
    }


def check_equilibrium(contract: ExperimentContract) -> dict[str, Any]:
    """Exactly gravity-balancing torque at rest keeps the arm at rest."""
    arm = contract.arm
    q = torch.tensor([0.7, -0.9], dtype=torch.float64)
    state = torch.cat([q, torch.zeros(2, dtype=torch.float64)])
    action = nominal.gravity_vector(q, arm)

    def accel(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return nominal.state_acceleration(s, a, arm)

    current = state.clone()
    for _ in range(40):
        current = rk4_transition(accel, current, action, contract.numerics.control_dt_s, 2)
    displacement = float(torch.max(torch.abs(current - state)))
    return {"max_displacement": displacement, "passed": displacement < 1e-9}


def check_reference_convergence(
    contract: ExperimentContract, count: int = 32
) -> dict[str, Any]:
    """One-step and long-horizon RK4 error against tight solve_ivp truth."""
    numerics = contract.numerics
    dt = numerics.control_dt_s
    states = _sample_states(contract, count)
    actions = _sample_actions(contract, count)

    def accel(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return nominal.state_acceleration(s, a, contract.arm)

    one_step: dict[int, np.ndarray] = {}
    for substeps in numerics.allowed_substep_candidates:
        stepped = rk4_transition(accel, states, actions, dt, substeps).numpy()
        errors = np.empty(count)
        for index in range(count):
            reference = _solve_ivp_transition(
                states[index].numpy(),
                actions[index].numpy(),
                dt,
                contract.arm,
                numerics.reference_rtol,
                numerics.reference_atol,
            )
            errors[index] = normalized_state_error(stepped[index] - reference)
        one_step[substeps] = errors

    # Eight-second free rollout (zero torque, damping on) from one state.
    long_state = torch.tensor([1.2, -0.5, 0.5, 0.8], dtype=torch.float64)
    zero_action = torch.zeros(2, dtype=torch.float64)
    horizon_steps = int(round(8.0 / dt))
    reference_long = _solve_ivp_transition(
        long_state.numpy(), zero_action.numpy(), 8.0, contract.arm,
        numerics.reference_rtol, numerics.reference_atol,
    )
    long_errors: dict[int, float] = {}
    for substeps in numerics.allowed_substep_candidates:
        current = long_state.clone()
        for _ in range(horizon_steps):
            current = rk4_transition(accel, current, zero_action, dt, substeps)
        long_errors[substeps] = float(normalized_state_error(current.numpy() - reference_long))

    gate = numerics.integration_gate
    per_substep: dict[str, Any] = {}
    for substeps, errors in one_step.items():
        p99 = float(np.quantile(errors, 0.99))
        maximum = float(np.max(errors))
        per_substep[str(substeps)] = {
            "one_step_p99": p99,
            "one_step_max": maximum,
            "eight_second_error": long_errors[substeps],
            "meets_gate": (
                p99 <= gate.one_step_normalized_error_p99_max
                and maximum <= gate.one_step_normalized_error_absolute_max
                and long_errors[substeps] <= gate.eight_second_normalized_error_absolute_max
            ),
        }
    smallest_passing = next(
        (s for s in numerics.allowed_substep_candidates if per_substep[str(s)]["meets_gate"]),
        None,
    )
    return {
        "per_substep": per_substep,
        "smallest_passing_substeps": smallest_passing,
        "passed": smallest_passing is not None,
        "_one_step_errors": {str(k): v.tolist() for k, v in one_step.items()},
    }


def check_float32_wrapper(contract: ExperimentContract, count: int = 256) -> dict[str, Any]:
    """Planning wrapper in float32 must track float64 within the gate."""
    numerics = contract.numerics
    states = _sample_states(contract, count)
    actions = _sample_actions(contract, count)

    def accel64(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return nominal.state_acceleration(s, a, contract.arm)

    def accel32(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return nominal.state_acceleration(s, a, contract.arm)

    stepped64 = rk4_transition(
        accel64, states, actions, numerics.control_dt_s, numerics.substeps_per_control_step
    ).numpy()
    stepped32 = rk4_transition(
        accel32,
        states.to(torch.float32),
        actions.to(torch.float32),
        numerics.control_dt_s,
        numerics.substeps_per_control_step,
    ).to(torch.float64).numpy()
    worst = float(np.max(normalized_state_error(stepped32 - stepped64)))
    limit = numerics.integration_gate.float32_vs_float64_one_step_normalized_error_max
    return {"max_normalized_deviation": worst, "limit": limit, "passed": worst <= limit}


def _write_plots(directory: Path, results: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energy = results["energy_conservation"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for substeps, trace in energy["_energy_traces"].items():
        initial = trace[0]
        axes[0].plot([abs(e - initial) for e in trace], label=f"{substeps} substep(s)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("control step")
    axes[0].set_ylabel("|E(t) - E(0)| [J]")
    axes[0].set_title("Conservative energy drift")
    axes[0].legend()

    axes[1].plot(results["dissipation"]["_energy_trace"])
    axes[1].set_xlabel("control step")
    axes[1].set_ylabel("T + V [J]")
    axes[1].set_title("Dissipative energy decay (u = 0)")
    figure.tight_layout()
    figure.savefig(directory / "energy.svg")
    plt.close(figure)

    convergence = results["reference_convergence"]
    figure, axis = plt.subplots(figsize=(5.5, 4))
    substep_counts = sorted(int(s) for s in convergence["per_substep"])
    p99 = [convergence["per_substep"][str(s)]["one_step_p99"] for s in substep_counts]
    axis.loglog(substep_counts, p99, marker="o")
    axis.set_xlabel("RK4 substeps per control interval")
    axis.set_ylabel("one-step normalized error (p99)")
    axis.set_title("Convergence toward solve_ivp reference")
    figure.tight_layout()
    figure.savefig(directory / "convergence.svg")
    plt.close(figure)


def run_simulator_verification(contract: ExperimentContract) -> dict[str, Any]:
    results: dict[str, Any] = {
        "mass_matrix": check_mass_matrix(contract),
        "batched_vs_reference": check_batched_matches_reference(contract),
        "gravity_gradient": check_gravity_gradient(contract),
        "energy_conservation": check_energy_conservation(contract),
        "dissipation": check_dissipation_monotonic(contract),
        "equilibrium": check_equilibrium(contract),
        "reference_convergence": check_reference_convergence(contract),
        "float32_wrapper": check_float32_wrapper(contract),
    }
    all_passed = all(section["passed"] for section in results.values())

    summary = {
        name: {k: v for k, v in section.items() if not k.startswith("_")}
        for name, section in results.items()
    }
    verification_id = content_id(
        "verification",
        {
            "suite_version": SUITE_VERSION,
            "root_seed": contract.numerics.root_seed,
            "control_dt_s": contract.numerics.control_dt_s,
            "arm": vars(contract.arm) | {},
        },
    )
    destination = verification_dir(verification_id)
    if is_complete(destination):
        verify_artifact(destination)
    else:
        import json

        def populate(directory: Path) -> None:
            (directory / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _write_plots(directory, results)

        write_artifact(
            destination,
            "simulator_verification",
            {"verification_id": verification_id, "suite_version": SUITE_VERSION},
            {"contract": contract.source_path},
            populate,
        )
    return {
        "all_passed": all_passed,
        "artifact": str(destination),
        "checks": summary,
    }
