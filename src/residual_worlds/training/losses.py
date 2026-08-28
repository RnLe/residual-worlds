"""Transition losses shared by every adapted method.

The one-step loss compares the model's integrated next state against
the observed one under fixed physical scales (angles / 1 rad,
velocities / 4 rad/s). Joint angles are unwrapped bounded coordinates
with hard stops, so the angle error is the raw coordinate difference --
a periodic error would incorrectly make the two opposite hard-stop
neighborhoods appear close.

The five-step loss unrolls the model open loop (no teacher forcing:
each prediction feeds the next) along recorded action sequences and
averages the same per-step error. The training objective is
``L1 + lambda5 * L5`` with both weights fixed for the whole study.

Only transitions are supervised; exact simulator accelerations never
appear here.
"""

from __future__ import annotations

import torch

from residual_worlds.models.base import Stepper
from residual_worlds.physics.integrators import AccelerationFn

ANGLE_SCALE = 1.0
VELOCITY_SCALE = 4.0


def normalized_state_loss(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Mean normalized squared state error over the batch (scalar)."""
    dq = (predicted[..., :2] - observed[..., :2]) / ANGLE_SCALE
    dqd = (predicted[..., 2:] - observed[..., 2:]) / VELOCITY_SCALE
    per_sample = 0.5 * torch.sum(dq**2, dim=-1) + 0.5 * torch.sum(dqd**2, dim=-1)
    return per_sample.mean()


def one_step_loss(
    acceleration: AccelerationFn,
    stepper: Stepper,
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
) -> torch.Tensor:
    predicted = stepper.step(acceleration, state, action)
    return normalized_state_loss(predicted, next_state)


def open_loop_loss(
    acceleration: AccelerationFn,
    stepper: Stepper,
    initial_state: torch.Tensor,
    actions: torch.Tensor,
    truth: torch.Tensor,
) -> torch.Tensor:
    """Average per-step error of an H-step open-loop rollout.

    Shapes: ``initial_state`` [B, 4], ``actions`` [B, H, 2], ``truth``
    [B, H, 4] (observed states after steps 1..H).
    """
    horizon = actions.shape[-2]
    state = initial_state
    losses = []
    for h in range(horizon):
        state = stepper.step(acceleration, state, actions[:, h])
        losses.append(normalized_state_loss(state, truth[:, h]))
    return torch.stack(losses).mean()


def combined_loss(
    acceleration: AccelerationFn,
    stepper: Stepper,
    one_step_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rollout_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    one_step_weight: float,
    rollout_weight: float,
) -> torch.Tensor:
    state, action, next_state = one_step_batch
    loss = one_step_weight * one_step_loss(acceleration, stepper, state, action, next_state)
    if rollout_batch is not None and rollout_weight > 0.0:
        initial, actions, truth = rollout_batch
        loss = loss + rollout_weight * open_loop_loss(
            acceleration, stepper, initial, actions, truth
        )
    return loss
