"""Common stepping wrapper shared by every planner-facing model.

A model is a continuous acceleration function; the ``Stepper`` turns it
into discrete transitions with the frozen control interval and RK4
substep count. All five conditions use the same wrapper, so a method
comparison can never become an integrator comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from residual_worlds.config import ExperimentContract
from residual_worlds.physics.integrators import (
    AccelerationFn,
    rk4_substep_endpoints,
    rk4_transition,
)


@dataclass(frozen=True)
class Stepper:
    control_dt_s: float
    substeps: int

    @staticmethod
    def from_contract(contract: ExperimentContract) -> Stepper:
        return Stepper(
            control_dt_s=contract.numerics.control_dt_s,
            substeps=contract.numerics.substeps_per_control_step,
        )

    def step(
        self, acceleration: AccelerationFn, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return rk4_transition(acceleration, state, action, self.control_dt_s, self.substeps)

    def substep_endpoints(
        self, acceleration: AccelerationFn, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return rk4_substep_endpoints(
            acceleration, state, action, self.control_dt_s, self.substeps
        )
