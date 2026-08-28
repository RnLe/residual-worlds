"""Black-box neural dynamics: an MLP predicts the complete acceleration.

Capacity-matched twin of the residual model: identical features, widths,
activation, output convention, and training procedure. Its only
structural difference from the residual condition is that nominal
physics is NOT added to its output, so it must represent gravity,
inertia, coupling, and actuation from data alone.
"""

from __future__ import annotations

import torch
from torch import nn

from residual_worlds.config import NeuralCommonConfig
from residual_worlds.models.normalization import (
    PhysicalScales,
    acceleration_from_network_output,
    features,
)

_ACTIVATIONS = {"silu": nn.SiLU}


def build_mlp(config: NeuralCommonConfig, generator: torch.Generator) -> nn.Sequential:
    """The shared MLP trunk, seeded explicitly (default PyTorch init rule)."""
    activation = _ACTIVATIONS[config.activation]
    layers: list[nn.Module] = []
    widths = (config.input_dim, *config.hidden_widths)
    for in_width, out_width in zip(widths[:-1], widths[1:], strict=True):
        layers.append(_seeded_linear(in_width, out_width, generator))
        layers.append(activation())
    layers.append(_seeded_linear(widths[-1], config.output_dim, generator))
    return nn.Sequential(*layers)


def _seeded_linear(in_width: int, out_width: int, generator: torch.Generator) -> nn.Linear:
    layer = nn.Linear(in_width, out_width)
    # Reproduce the default Kaiming-uniform/bias rule from an explicit
    # generator so member initializations are attributable to seeds.
    bound_w = (1.0 / in_width) ** 0.5 * 3.0**0.5
    with torch.no_grad():
        layer.weight.uniform_(-bound_w, bound_w, generator=generator)
        bound_b = (1.0 / in_width) ** 0.5
        layer.bias.uniform_(-bound_b, bound_b, generator=generator)
    return layer


class BlackBoxModel(nn.Module):
    model_id = "blackbox"

    def __init__(
        self,
        config: NeuralCommonConfig,
        scales: PhysicalScales,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.network = build_mlp(config, generator)
        self._scales = scales

    def acceleration(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        phi = features(state, action, self._scales)
        return acceleration_from_network_output(self.network(phi), self._scales)

    forward = acceleration
