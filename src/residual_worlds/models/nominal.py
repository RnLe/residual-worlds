"""The unchanged nominal model: analytical physics, no adaptation data.

This condition is never fitted, corrected, or tuned; it measures how far
the original approximate equations carry control in a changed world.
"""

from __future__ import annotations

import torch

from residual_worlds.physics.nominal import state_acceleration
from residual_worlds.types import ArmParameters


class NominalModel:
    model_id = "nominal"

    def __init__(self, arm: ArmParameters) -> None:
        self._arm = arm

    def acceleration(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state_acceleration(state, action, self._arm)
