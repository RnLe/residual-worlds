"""Read access to immutable datasets: budgets, batches, and segments.

Loaders enforce the information boundary mechanically: the returned
view exposes transitions and split structure only. Hidden world
parameters live in the artifact manifest's provenance section, which
this module deliberately never returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from residual_worlds.provenance import read_manifest


@dataclass(frozen=True)
class DatasetView:
    """One immutable dataset, loaded for training or fitting."""

    directory: Path
    state: np.ndarray  # [N, 4] float64
    action: np.ndarray  # [N, 2] float64
    next_state: np.ndarray  # [N, 4] float64
    unit_id: np.ndarray  # [N] int64
    trajectory_id: np.ndarray  # [N] int64
    step_index: np.ndarray  # [N] int32
    component_code: np.ndarray  # [N] int16
    train_units: tuple[int, ...]
    validation_units: tuple[int, ...]

    def units_for_budget(
        self, budget: int, unit_size: int, train_fraction: float
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Nested-prefix unit ids for one budget: (train, validation)."""
        units = budget // unit_size
        train_count = round(units * train_fraction)
        validation_count = units - train_count
        if train_count > len(self.train_units) or validation_count > len(
            self.validation_units
        ):
            raise ValueError(f"budget {budget} exceeds this dataset")
        return (
            self.train_units[:train_count],
            self.validation_units[:validation_count],
        )

    def rows_for_units(self, units: tuple[int, ...]) -> np.ndarray:
        """Sorted transition row indices belonging to the given units."""
        mask = np.isin(self.unit_id, np.asarray(units, dtype=np.int64))
        return np.nonzero(mask)[0]

    def rollout_origins(self, rows: np.ndarray, horizon: int) -> np.ndarray:
        """Origins (subset of ``rows``) with ``horizon`` consecutive steps.

        A window is eligible only if all its rows are in ``rows``, share
        one trajectory, and have consecutive step indices -- windows
        never cross a reset or a unit boundary.
        """
        row_set = set(int(r) for r in rows)
        origins = []
        for origin in rows:
            end = int(origin) + horizon - 1
            if end >= self.state.shape[0]:
                continue
            window = range(int(origin), end + 1)
            if not all(r in row_set for r in window):
                continue
            trajectory = self.trajectory_id[int(origin) : end + 1]
            steps = self.step_index[int(origin) : end + 1]
            if (trajectory == trajectory[0]).all() and (np.diff(steps) == 1).all():
                origins.append(int(origin))
        return np.asarray(origins, dtype=np.int64)

    def tensors(
        self, rows: np.ndarray, dtype: torch.dtype = torch.float64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.state[rows]).to(dtype),
            torch.from_numpy(self.action[rows]).to(dtype),
            torch.from_numpy(self.next_state[rows]).to(dtype),
        )

    def membership_digest(self, units: tuple[int, ...]) -> str:
        """Hash of the exact transition membership (pairing audits)."""
        import hashlib

        rows = self.rows_for_units(units)
        return hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()


def load_dataset(directory: Path) -> DatasetView:
    read_manifest(directory)  # requires COMPLETE and a valid manifest
    with np.load(directory / "transitions.npz") as archive:
        state = archive["state"]
        action = archive["action"]
        next_state = archive["next_state"]
        unit_id = archive["collection_unit_id"]
        trajectory_id = archive["trajectory_id"]
        step_index = archive["step_index"]
        component_code = archive["component_code"]
    split_path = directory / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_units = tuple(split["train_units"])
        validation_units = tuple(split["validation_units"])
    else:
        train_units = ()
        validation_units = ()
    return DatasetView(
        directory=directory,
        state=state,
        action=action,
        next_state=next_state,
        unit_id=unit_id,
        trajectory_id=trajectory_id,
        step_index=step_index,
        component_code=component_code,
        train_units=train_units,
        validation_units=validation_units,
    )


def load_segment_registry(directory: Path) -> dict[int, np.ndarray]:
    """Per-horizon eligible origin rows of a prediction set."""
    table = pq.read_table(directory / "segment_registry.parquet")
    horizon = table["horizon"].to_numpy()
    origin = table["origin_row"].to_numpy()
    return {
        int(h): origin[horizon == h].astype(np.int64) for h in np.unique(horizon)
    }
