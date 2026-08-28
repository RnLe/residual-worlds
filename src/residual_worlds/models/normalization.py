"""Fixed physical feature and output scales.

No empirical (data-derived) normalization is permitted anywhere in this
study: scales are declared physical constants, identical for every
method, budget, and pipeline, so a large-budget statistic can never leak
into a small-budget model through a fitted normalizer.

Feature vector (dimension 8):

    phi(x, u) = [sin q1, cos q1, sin q2, cos q2,
                 qd1 / s_qd, qd2 / s_qd, u1 / u1_max, u2 / u2_max]

Network outputs are accelerations divided by fixed per-joint scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from residual_worlds.config import ExperimentContract


@dataclass(frozen=True)
class PhysicalScales:
    velocity_scale_rad_s: float
    torque_limit_nm: tuple[float, float]
    acceleration_scale_rad_s2: tuple[float, float]

    @staticmethod
    def from_contract(contract: ExperimentContract) -> PhysicalScales:
        return PhysicalScales(
            velocity_scale_rad_s=contract.data.velocity_scale_rad_s,
            torque_limit_nm=contract.arm.torque_limit_nm,
            acceleration_scale_rad_s2=contract.data.acceleration_scale_rad_s2,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "velocity_scale_rad_s": self.velocity_scale_rad_s,
            "torque_limit_nm": list(self.torque_limit_nm),
            "acceleration_scale_rad_s2": list(self.acceleration_scale_rad_s2),
        }


def features(state: torch.Tensor, action: torch.Tensor, scales: PhysicalScales) -> torch.Tensor:
    """The shared eight-dimensional model input phi(x, u), ``[..., 8]``."""
    q1, q2 = state[..., 0], state[..., 1]
    qd = state[..., 2:] / scales.velocity_scale_rad_s
    limit = torch.as_tensor(scales.torque_limit_nm, dtype=state.dtype, device=state.device)
    u = action / limit
    return torch.stack(
        (
            torch.sin(q1),
            torch.cos(q1),
            torch.sin(q2),
            torch.cos(q2),
            qd[..., 0],
            qd[..., 1],
            u[..., 0],
            u[..., 1],
        ),
        dim=-1,
    )


def acceleration_from_network_output(
    output: torch.Tensor, scales: PhysicalScales
) -> torch.Tensor:
    """Undo the fixed output scaling: network units -> rad/s^2."""
    scale = torch.as_tensor(
        scales.acceleration_scale_rad_s2, dtype=output.dtype, device=output.device
    )
    return output * scale
