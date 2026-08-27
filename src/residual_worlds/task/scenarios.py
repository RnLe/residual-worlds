"""Stratified scenario generation with structural feasibility proofs.

Each scenario is one frozen task instance: an initial state, three
ordered targets, and one circular obstacle. Scenarios are stratified
over twelve *joint structural strata* -- six visit orders of the
(left, middle, right) workspace sectors crossed with the obstacle lying
on the first or second target-to-target chord -- assigned by a seeded
rotation so their correlation can never become a hidden factor.

Acceptance is purely structural and method-blind: analytic two-link IK
(both elbow branches) must reach every target inside hard-limit margins
and obstacle clearance, and a coarse joint-space grid graph must
contain a collision-free path from the initial pose through the three
targets in visit order. The graph proves geometric reachability only;
it never calls a controller, a learned model, or target dynamics, so no
outcome can influence which scenarios exist.

Generation is deterministic: bank and index fix the random stream, and
exhausting the attempt budget is a hard error, not a silent fallback.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from residual_worlds.config import ExperimentContract, ScenarioGeneratorConfig
from residual_worlds.identity import content_id
from residual_worlds.physics.kinematics import end_effector_position
from residual_worlds.seeds import numpy_generator, seed_record
from residual_worlds.task.geometry import arm_clearance_with_radius
from residual_worlds.types import ArmParameters, Scenario

SECTOR_NAMES = ("left", "middle", "right")
SECTOR_ORDERS: tuple[tuple[int, int, int], ...] = tuple(
    (a, b, c) for a, b, c in itertools.permutations((0, 1, 2))
)

# Conservative clearance used for every structural-feasibility check:
# links must clear the obstacle (plus arm safety radius) by this margin.
_STRUCTURAL_CLEARANCE_M = 0.02


class ScenarioGenerationError(RuntimeError):
    pass


def stratum_assignment(root_seed: int, bank: str, count: int) -> list[int]:
    """Seeded rotation over the twelve joint strata for one bank."""
    rng = numpy_generator(root_seed, "scenario_strata", bank)
    rotation = int(rng.integers(0, 12))
    return [(rotation + index) % 12 for index in range(count)]


def _two_link_ik(
    target: np.ndarray, arm: ArmParameters
) -> list[np.ndarray]:
    """Analytic IK; returns 0-2 joint solutions (elbow-down and elbow-up)."""
    l1, l2 = arm.link_lengths_m
    r2 = float(target[0] ** 2 + target[1] ** 2)
    cos_q2 = (r2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    if abs(cos_q2) > 1.0:
        return []
    solutions = []
    for sign in (1.0, -1.0):
        q2 = sign * math.acos(max(-1.0, min(1.0, cos_q2)))
        q1 = math.atan2(target[1], target[0]) - math.atan2(
            l2 * math.sin(q2), l1 + l2 * math.cos(q2)
        )
        # Wrap q1 into (-pi, pi]; the joint range is inside that interval.
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        solutions.append(np.array([q1, q2], dtype=np.float64))
    return solutions


def _within_limits(q: np.ndarray, margin: float, arm: ArmParameters) -> bool:
    return all(
        arm.q_min_rad[j] + margin <= q[j] <= arm.q_max_rad[j] - margin for j in range(2)
    )


def _clearance(
    q: np.ndarray | torch.Tensor,
    center: np.ndarray,
    radius: float,
    arm: ArmParameters,
    safety_radius: float,
) -> torch.Tensor:
    q_tensor = torch.as_tensor(np.asarray(q), dtype=torch.float64)
    return arm_clearance_with_radius(
        q_tensor,
        torch.tensor(center, dtype=torch.float64),
        radius,
        safety_radius,
        arm,
    )


def _segment_configs_clear(
    q_a: np.ndarray,
    q_b: np.ndarray,
    center: np.ndarray,
    radius: float,
    arm: ArmParameters,
    safety_radius: float,
    limit_margin: float,
    resolution_m: float,
) -> bool:
    """Sampled check of the straight joint-space segment q_a -> q_b."""
    l1, l2 = arm.link_lengths_m
    dq = np.abs(q_b - q_a)
    bound = (l1 + l2) * dq[0] + l2 * dq[1]
    samples = max(2, int(math.ceil(bound / resolution_m)) + 1)
    s = np.linspace(0.0, 1.0, samples)
    path = q_a[None, :] * (1.0 - s[:, None]) + q_b[None, :] * s[:, None]
    for j in range(2):
        low, high = path[:, j].min(), path[:, j].max()
        if low < arm.q_min_rad[j] + limit_margin or high > arm.q_max_rad[j] - limit_margin:
            return False
    clearance = _clearance(path, center, radius, arm, safety_radius)
    return bool((clearance >= _STRUCTURAL_CLEARANCE_M).all())


class _GridGraph:
    """Coarse joint-space occupancy graph for structural reachability."""

    def __init__(
        self,
        generator: ScenarioGeneratorConfig,
        center: np.ndarray,
        radius: float,
        arm: ArmParameters,
        safety_radius: float,
    ) -> None:
        self._arm = arm
        self._center = center
        self._radius = radius
        self._safety_radius = safety_radius
        self._resolution = generator.edge_resolution_bound_m
        self._margin = generator.ik_hard_limit_margin_rad
        self.q1_values = np.linspace(
            arm.q_min_rad[0], arm.q_max_rad[0], generator.q1_grid_points
        )
        self.q2_values = np.linspace(
            arm.q_min_rad[1], arm.q_max_rad[1], generator.q2_grid_points
        )
        grid_q1, grid_q2 = np.meshgrid(self.q1_values, self.q2_values, indexing="ij")
        configs = np.stack([grid_q1, grid_q2], axis=-1)  # [N1, N2, 2]
        clearance = _clearance(configs, center, radius, arm, safety_radius).numpy()
        margin_ok = (
            (grid_q1 >= arm.q_min_rad[0] + self._margin)
            & (grid_q1 <= arm.q_max_rad[0] - self._margin)
            & (grid_q2 >= arm.q_min_rad[1] + self._margin)
            & (grid_q2 <= arm.q_max_rad[1] - self._margin)
        )
        self.valid = (clearance >= _STRUCTURAL_CLEARANCE_M) & margin_ok
        self._configs = configs
        self._labels: np.ndarray | None = None
        self._edge_ok: dict[tuple[int, int], np.ndarray] = {}
        for direction in ((1, 0), (0, 1), (1, 1), (1, -1)):
            self._edge_ok[direction] = self._edge_validity(direction)

    def _edge_validity(self, direction: tuple[int, int]) -> np.ndarray:
        """Validity of edges from each node toward ``direction`` (sampled).

        Directions are restricted to di >= 0; the reverse edge shares
        the same table. Entry [i, j] of the returned array refers to the
        edge whose source node is (i + i_offset, j + j_offset) with
        offsets (0, max(0, -dj)).
        """
        di, dj = direction
        n1, n2 = self.valid.shape
        i0, i1 = 0, n1 - di
        j0, j1 = max(0, -dj), n2 - max(0, dj)
        source = self._configs[i0:i1, j0:j1]
        target = self._configs[i0 + di : i1 + di, j0 + dj : j1 + dj]
        delta = np.abs(target - source)  # constant per direction class
        l1, l2 = self._arm.link_lengths_m
        bound = (l1 + l2) * float(delta[..., 0].max(initial=0.0)) + l2 * float(
            delta[..., 1].max(initial=0.0)
        )
        samples = max(2, int(math.ceil(bound / self._resolution)) + 1)
        s = np.linspace(0.0, 1.0, samples).reshape(-1, 1, 1, 1)
        path = source[None, ...] * (1.0 - s) + target[None, ...] * s
        clearance = _clearance(
            path, self._center, self._radius, self._arm, self._safety_radius
        ).numpy()
        ok: np.ndarray = (clearance >= _STRUCTURAL_CLEARANCE_M).all(axis=0)
        return ok

    def _component_labels(self) -> np.ndarray:
        """Connected-component label per grid node (vectorized, cached)."""
        if self._labels is not None:
            return self._labels
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        n1, n2 = self.valid.shape
        total = n1 * n2
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        for (di, dj), table in self._edge_ok.items():
            i0, j0 = 0, max(0, -dj)
            si, sj = np.nonzero(table)
            source_i, source_j = si + i0, sj + j0
            target_i, target_j = source_i + di, source_j + dj
            keep = self.valid[source_i, source_j] & self.valid[target_i, target_j]
            rows.append((source_i * n2 + source_j)[keep])
            cols.append((target_i * n2 + target_j)[keep])
        row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
        col = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
        data = np.ones(row.shape[0], dtype=np.int8)
        graph = coo_matrix((data, (row, col)), shape=(total, total))
        _count, labels = connected_components(graph, directed=False)
        self._labels = labels.reshape(n1, n2)
        return self._labels

    def same_component(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        labels = self._component_labels()
        return bool(labels[a] == labels[b])

    def nearest_valid_node(self, q: np.ndarray) -> tuple[int, int] | None:
        if not self.valid.any():
            return None
        i = int(np.abs(self.q1_values - q[0]).argmin())
        j = int(np.abs(self.q2_values - q[1]).argmin())
        # Search outward in growing rings for the nearest valid node.
        n1, n2 = self.valid.shape
        best: tuple[int, int] | None = None
        best_distance = float("inf")
        for radius in range(0, max(n1, n2)):
            i_lo, i_hi = max(0, i - radius), min(n1, i + radius + 1)
            j_lo, j_hi = max(0, j - radius), min(n2, j + radius + 1)
            window = self.valid[i_lo:i_hi, j_lo:j_hi]
            if window.any():
                indices = np.argwhere(window)
                for di, dj in indices:
                    node = (i_lo + int(di), j_lo + int(dj))
                    dist = math.hypot(
                        self.q1_values[node[0]] - q[0], self.q2_values[node[1]] - q[1]
                    )
                    if dist < best_distance:
                        best_distance = dist
                        best = node
                if best is not None:
                    return best
        return best

    def node_config(self, node: tuple[int, int]) -> np.ndarray:
        return np.array(
            [self.q1_values[node[0]], self.q2_values[node[1]]], dtype=np.float64
        )

    def connect_pose(self, q: np.ndarray) -> tuple[int, int] | None:
        """Attach an exact pose to its nearest valid node if the link is clear."""
        node = self.nearest_valid_node(q)
        if node is None:
            return None
        if _segment_configs_clear(
            q,
            self.node_config(node),
            self._center,
            self._radius,
            self._arm,
            self._safety_radius,
            self._margin,
            self._resolution,
        ):
            return node
        return None

def _attempt_scenario(
    contract: ExperimentContract,
    rng: np.random.Generator,
    stratum_id: int,
) -> Scenario | None:
    """One seeded generation attempt; None means a constraint failed."""
    generator = contract.task.scenario_generator
    arm = contract.arm
    order = SECTOR_ORDERS[stratum_id // 2]
    chord_index = stratum_id % 2

    # Initial state.
    q1 = rng.uniform(*generator.initial_q1_rad)
    q2 = rng.uniform(*generator.initial_q2_rad)
    qd1 = rng.uniform(*generator.initial_qd_rad_s)
    qd2 = rng.uniform(*generator.initial_qd_rad_s)
    initial_q = np.array([q1, q2], dtype=np.float64)
    if not _within_limits(initial_q, generator.hard_joint_margin_rad, arm):
        return None

    # One target per sector, then the stratum's visit order.
    sector_targets: list[np.ndarray] = []
    for sector in SECTOR_NAMES:
        x = rng.uniform(*generator.x_sectors_m[sector])
        y = rng.uniform(*generator.y_m)
        target = np.array([x, y], dtype=np.float64)
        if not (
            generator.radial_distance_m[0]
            <= float(np.linalg.norm(target))
            <= generator.radial_distance_m[1]
        ):
            return None
        sector_targets.append(target)
    ordered = [sector_targets[sector] for sector in order]

    for a, b in itertools.combinations(ordered, 2):
        if float(np.linalg.norm(a - b)) < generator.minimum_pairwise_separation_m:
            return None
    initial_ee = end_effector_position(
        torch.tensor(initial_q, dtype=torch.float64), arm
    ).numpy()
    if (
        float(np.linalg.norm(ordered[0] - initial_ee))
        < generator.minimum_initial_to_first_distance_m
    ):
        return None
    chord_length = sum(
        float(np.linalg.norm(b - a)) for a, b in zip(ordered[:-1], ordered[1:], strict=True)
    )
    low, high = generator.total_ordered_chord_length_m
    if not (low <= chord_length <= high):
        return None

    # Obstacle on the assigned chord.
    chord_a, chord_b = ordered[chord_index], ordered[chord_index + 1]
    lam = rng.uniform(*generator.obstacle_chord_fraction)
    radius = rng.uniform(*generator.obstacle_radius_m)
    offset_fraction = rng.uniform(*generator.obstacle_perpendicular_offset_in_radii)
    chord = chord_b - chord_a
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm < 1e-9:
        return None
    normal = np.array([-chord[1], chord[0]]) / chord_norm
    center = chord_a + lam * chord + offset_fraction * radius * normal
    if not (
        generator.obstacle_center_x_m[0] <= center[0] <= generator.obstacle_center_x_m[1]
        and generator.obstacle_center_y_m[0] <= center[1] <= generator.obstacle_center_y_m[1]
    ):
        return None
    min_target_clearance = radius + generator.minimum_target_center_clearance_beyond_radius_m
    if any(float(np.linalg.norm(t - center)) < min_target_clearance for t in ordered):
        return None
    initial_clearance = float(
        _clearance(initial_q, center, radius, arm, contract.task.arm_safety_radius_m)
    )
    if initial_clearance < generator.minimum_signed_obstacle_clearance_m:
        return None

    # Analytic IK, both branches, with margins and obstacle clearance.
    retained_branches: list[list[int]] = []
    branch_configs: list[list[np.ndarray]] = []
    for target in ordered:
        valid_branches: list[int] = []
        configs: list[np.ndarray] = []
        for branch, solution in enumerate(_two_link_ik(target, arm)):
            if not _within_limits(solution, generator.ik_hard_limit_margin_rad, arm):
                continue
            clearance = float(
                _clearance(solution, center, radius, arm, contract.task.arm_safety_radius_m)
            )
            if clearance < _STRUCTURAL_CLEARANCE_M:
                continue
            valid_branches.append(branch)
            configs.append(solution)
        if not valid_branches:
            return None
        retained_branches.append(valid_branches)
        branch_configs.append(configs)

    # Joint-space grid graph: the initial pose and at least one IK branch
    # of every target must lie in one connected component.
    graph = _GridGraph(generator, center, radius, arm, contract.task.arm_safety_radius_m)
    start = graph.connect_pose(initial_q)
    if start is None:
        return None
    for configs in branch_configs:
        if not any(
            (node := graph.connect_pose(config)) is not None
            and graph.same_component(start, node)
            for config in configs
        ):
            return None

    payload: dict[str, Any] = {
        "generator_version": generator.version,
        "stratum_id": stratum_id,
        "target_order": list(order),
        "obstacle_chord_index": chord_index,
        "initial_state": [q1, q2, qd1, qd2],
        "targets_xy_m": [[float(t[0]), float(t[1])] for t in ordered],
        "obstacle_xy_radius_m": [float(center[0]), float(center[1]), float(radius)],
        "timeout_steps": contract.task.timeout_steps,
    }
    scenario_id = content_id("scenario", payload)
    return Scenario(
        scenario_id=scenario_id,
        bank="",  # filled by generate_bank
        index=-1,
        stratum_id=stratum_id,
        target_order=(order[0], order[1], order[2]),
        obstacle_chord_index=chord_index,
        initial_state=(q1, q2, qd1, qd2),
        targets_xy_m=tuple((float(t[0]), float(t[1])) for t in ordered),
        obstacle_xy_radius_m=(float(center[0]), float(center[1]), float(radius)),
        timeout_steps=contract.task.timeout_steps,
    )


def generate_scenario(
    contract: ExperimentContract, bank: str, index: int, stratum_id: int
) -> tuple[Scenario, int]:
    """Generate one scenario deterministically; returns (scenario, attempts)."""
    generator = contract.task.scenario_generator
    rng = numpy_generator(contract.numerics.root_seed, "scenario", bank, index)
    for attempt in range(1, generator.maximum_attempts_per_scenario + 1):
        scenario = _attempt_scenario(contract, rng, stratum_id)
        if scenario is not None:
            from dataclasses import replace

            return replace(scenario, bank=bank, index=index), attempt
    raise ScenarioGenerationError(
        f"scenario generation exhausted {generator.maximum_attempts_per_scenario} attempts "
        f"for bank={bank} index={index} stratum={stratum_id}"
    )


def generate_bank(contract: ExperimentContract, bank: str) -> list[Scenario]:
    generator = contract.task.scenario_generator
    if bank not in generator.bank_counts:
        raise ValueError(f"unknown scenario bank {bank!r}")
    count = generator.bank_counts[bank]
    strata = stratum_assignment(contract.numerics.root_seed, bank, count)
    scenarios: list[Scenario] = []
    for index in range(count):
        scenario, _attempts = generate_scenario(contract, bank, index, strata[index])
        scenarios.append(scenario)
    return scenarios


def write_bank_manifest(
    contract: ExperimentContract, bank: str, scenarios: list[Scenario], output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = seed_record(contract.numerics.root_seed, "scenario_strata", bank)
    document = {
        "schema": 1,
        "bank": bank,
        "generator_version": contract.task.scenario_generator.version,
        "root_seed": contract.numerics.root_seed,
        "strata_seed_digest": seed.digest_sha256,
        "scenarios": [
            {
                "scenario_id": s.scenario_id,
                "index": s.index,
                "stratum_id": s.stratum_id,
                "target_order": list(s.target_order),
                "obstacle_chord_index": s.obstacle_chord_index,
                "initial_state": list(s.initial_state),
                "targets_xy_m": [list(t) for t in s.targets_xy_m],
                "obstacle_xy_radius_m": list(s.obstacle_xy_radius_m),
                "timeout_steps": s.timeout_steps,
            }
            for s in scenarios
        ],
    }
    document["bank_content_id"] = content_id("scenario", {"bank_document": document})
    path = output_dir / f"{bank}.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_bank(path: Path) -> list[Scenario]:
    document = json.loads(path.read_text(encoding="utf-8"))
    scenarios = []
    for entry in document["scenarios"]:
        scenarios.append(
            Scenario(
                scenario_id=entry["scenario_id"],
                bank=document["bank"],
                index=entry["index"],
                stratum_id=entry["stratum_id"],
                target_order=tuple(entry["target_order"]),
                obstacle_chord_index=entry["obstacle_chord_index"],
                initial_state=tuple(entry["initial_state"]),
                targets_xy_m=tuple(tuple(t) for t in entry["targets_xy_m"]),
                obstacle_xy_radius_m=tuple(entry["obstacle_xy_radius_m"]),
                timeout_steps=entry["timeout_steps"],
            )
        )
    return scenarios
