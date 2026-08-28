"""Bootstrap ensembles of neural dynamics models.

Three deterministic bootstrap members per neural method. The ensemble
is a practical variance diagnostic and a mean-cost planning device --
never a calibrated posterior. The planner rolls every candidate through
each member separately and averages member *costs*; it must not average
accelerations first, because a nonlinear rollout of an average model is
not the average of the member rollouts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from residual_worlds.physics.integrators import AccelerationFn


class EnsembleModel:
    """Ordered collection of member models sharing one interface."""

    def __init__(self, model_id: str, members: Sequence[AccelerationFn]) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.model_id = model_id
        self.members = tuple(members)

    def __len__(self) -> int:
        return len(self.members)

    def member_accelerations(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Stacked member outputs, shape ``[members, ..., 2]``."""
        return torch.stack([member(state, action) for member in self.members], dim=0)

    def mean_acceleration(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Diagnostic mean; not used for candidate scoring in the planner."""
        return self.member_accelerations(state, action).mean(dim=0)

    def spread(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Member disagreement (population standard deviation), ``[..., 2]``."""
        return self.member_accelerations(state, action).std(dim=0, correction=0)
