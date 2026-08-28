"""Residual dynamics: nominal physics plus a learned acceleration correction.

    qdd_hat = f0(x, u) + r_theta(phi(x, u))

The correction network is the same MLP as the black box (equal trainable
parameter count, same features), with one deliberate asymmetry that IS
the tested inductive bias: its final layer starts at exactly zero, so
the untrained model reproduces nominal physics bit-for-bit. The learned
object is an acceleration correction over the visited state-action
distribution -- not a uniquely identified missing force.
"""

from __future__ import annotations

import torch
from torch import nn

from residual_worlds.config import NeuralCommonConfig
from residual_worlds.models.black_box import build_mlp
from residual_worlds.models.normalization import (
    PhysicalScales,
    acceleration_from_network_output,
    features,
)
from residual_worlds.physics.nominal import state_acceleration
from residual_worlds.types import ArmParameters


class ResidualModel(nn.Module):
    model_id = "residual"

    def __init__(
        self,
        config: NeuralCommonConfig,
        scales: PhysicalScales,
        arm: ArmParameters,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.network = build_mlp(config, generator)
        # Zero-initialize the output head: the prior starts exactly nominal.
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()
        self._scales = scales
        self._arm = arm

    def correction(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """The learned acceleration correction r_theta alone, ``[..., 2]``."""
        phi = features(state, action, self._scales)
        return acceleration_from_network_output(self.network(phi), self._scales)

    def acceleration(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state_acceleration(state, action, self._arm) + self.correction(state, action)

    forward = acceleration
