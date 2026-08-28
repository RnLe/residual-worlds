"""Excitation signals for target-world data collection.

Three components with fixed proportions:

* band-limited random torque -- seeded Gaussian at the control rate
  through a fourth-order Butterworth low-pass with a seeded burn-in,
  normalized per channel by its pre-clip maximum, scaled to the
  exploration envelope; broad local excitation without white-noise jerk;
* phase-randomized multisine -- fixed frequency comb with seeded
  independent phases per joint, deterministically scaled to the same
  envelope; persistent excitation across frequencies;
* nominal-MPC task rollouts plus a small held piecewise-constant
  perturbation -- task-relevant states and action patterns (assembled
  in ``generate.py``, which owns the controller; only the perturbation
  signal lives here).

Command sequences are generated up front from the unit's seed and are
never resampled in response to the trajectory; a reset simply continues
consuming the same sequence.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

from residual_worlds.config import ExplorationConfig


def band_limited_random_commands(
    rng: np.random.Generator, count: int, config: ExplorationConfig
) -> np.ndarray:
    """Filtered random torque, shape ``[count, 2]``, clipped to the envelope."""
    total = count + config.random_burn_in_samples
    raw = rng.standard_normal(size=(total, 2))
    b, a = butter(
        4, config.random_cutoff_hz, fs=config.random_sample_rate_hz, btype="low"
    )
    filtered = lfilter(b, a, raw, axis=0)[config.random_burn_in_samples :]
    envelope = np.asarray(config.torque_envelope_nm)
    peak = np.abs(filtered).max(axis=0, keepdims=True)
    peak = np.where(peak > 0.0, peak, 1.0)
    scaled = filtered / peak * envelope
    clipped: np.ndarray = np.clip(scaled, -envelope, envelope)
    return clipped


def multisine_commands(
    rng: np.random.Generator, count: int, config: ExplorationConfig, control_dt_s: float
) -> np.ndarray:
    """Phase-randomized multisine torque, shape ``[count, 2]``."""
    t = np.arange(count) * control_dt_s
    frequencies = np.asarray(config.multisine_frequencies_hz)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(2, frequencies.shape[0]))
    signal = np.zeros((count, 2))
    for joint in range(2):
        signal[:, joint] = np.sin(
            2.0 * np.pi * frequencies[None, :] * t[:, None] + phases[joint][None, :]
        ).sum(axis=1)
    envelope = np.asarray(config.torque_envelope_nm)
    peak = np.abs(signal).max(axis=0, keepdims=True)
    peak = np.where(peak > 0.0, peak, 1.0)
    clipped: np.ndarray = np.clip(signal / peak * envelope, -envelope, envelope)
    return clipped


def perturbation_commands(
    rng: np.random.Generator, count: int, config: ExplorationConfig
) -> np.ndarray:
    """Held piecewise-constant uniform perturbation for MPC units."""
    hold = config.mpc_perturbation_hold_steps
    segments = -(-count // hold)
    values = rng.uniform(
        config.mpc_perturbation_low_nm, config.mpc_perturbation_high_nm, size=(segments, 2)
    )
    return np.repeat(values, hold, axis=0)[:count]
