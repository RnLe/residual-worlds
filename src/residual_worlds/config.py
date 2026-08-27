"""Strict experiment-contract loading.

The YAML contract is parsed exactly once, validated, and turned into
frozen dataclasses. Three rules keep configuration honest:

* duplicate keys are a parse error (plain ``yaml.safe_load`` silently
  keeps the last one, which has bitten me before);
* unknown keys are an error -- a typo cannot silently deactivate a
  scientific setting;
* raw dictionaries do not flow into physics, training, or planning
  code; consumers receive typed frozen values.

Semantic cross-checks (budget arithmetic, world component equality,
profile consistency, contrast weights) run here as well, so an invalid
contract fails at load time rather than mid-experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from residual_worlds.types import ArmParameters


class ContractError(ValueError):
    """Raised when the experiment contract is malformed or inconsistent."""


# ---------------------------------------------------------------------------
# Strict YAML parsing


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            raise ContractError(f"duplicate YAML key {key!r} at {key_node.start_mark}")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_strict_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.load(handle, Loader=_StrictLoader)  # noqa: S506 - strict subclass
    if not isinstance(document, dict):
        raise ContractError(f"{path}: top level must be a mapping")
    return document


# ---------------------------------------------------------------------------
# Guided traversal with unknown-key detection


class _Node:
    """A mapping wrapper that tracks which keys were consumed."""

    def __init__(self, mapping: dict[str, Any], path: str) -> None:
        self._mapping = mapping
        self._path = path
        self._consumed: set[str] = set()

    def _get(self, key: str) -> Any:
        if key not in self._mapping:
            raise ContractError(f"missing key {self._path}.{key}")
        self._consumed.add(key)
        return self._mapping[key]

    def has(self, key: str) -> bool:
        return key in self._mapping

    def child(self, key: str) -> _Node:
        value = self._get(key)
        if not isinstance(value, dict):
            raise ContractError(f"{self._path}.{key} must be a mapping")
        return _Node(value, f"{self._path}.{key}")

    def raw(self, key: str) -> Any:
        """Consume a subtree without typed validation (presentation config)."""
        return self._get(key)

    def str_(self, key: str) -> str:
        value = self._get(key)
        if not isinstance(value, str):
            raise ContractError(f"{self._path}.{key} must be a string")
        return value

    def bool_(self, key: str) -> bool:
        value = self._get(key)
        if not isinstance(value, bool):
            raise ContractError(f"{self._path}.{key} must be a boolean")
        return value

    def int_(self, key: str, minimum: int | None = None) -> int:
        value = self._get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{self._path}.{key} must be an integer")
        if minimum is not None and value < minimum:
            raise ContractError(f"{self._path}.{key} must be >= {minimum}")
        return value

    def float_(self, key: str, minimum: float | None = None, positive: bool = False) -> float:
        value = self._get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"{self._path}.{key} must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise ContractError(f"{self._path}.{key} must be finite")
        if positive and result <= 0:
            raise ContractError(f"{self._path}.{key} must be > 0")
        if minimum is not None and result < minimum:
            raise ContractError(f"{self._path}.{key} must be >= {minimum}")
        return result

    def str_list(self, key: str) -> tuple[str, ...]:
        value = self._get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ContractError(f"{self._path}.{key} must be a list of strings")
        return tuple(value)

    def int_list(self, key: str) -> tuple[int, ...]:
        value = self._get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            raise ContractError(f"{self._path}.{key} must be a list of integers")
        return tuple(value)

    def float_list(self, key: str) -> tuple[float, ...]:
        value = self._get(key)
        if not isinstance(value, list):
            raise ContractError(f"{self._path}.{key} must be a list of numbers")
        result: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ContractError(f"{self._path}.{key} must be a list of numbers")
            number = float(item)
            if not math.isfinite(number):
                raise ContractError(f"{self._path}.{key} contains a non-finite number")
            result.append(number)
        return tuple(result)

    def float_pair(self, key: str, positive: bool = False) -> tuple[float, float]:
        values = self.float_list(key)
        if len(values) != 2:
            raise ContractError(f"{self._path}.{key} must have exactly two entries")
        if positive and any(value <= 0 for value in values):
            raise ContractError(f"{self._path}.{key} entries must be > 0")
        return (values[0], values[1])

    def float_range(self, key: str) -> tuple[float, float]:
        low, high = self.float_pair(key)
        if low >= high:
            raise ContractError(f"{self._path}.{key} must be an increasing [low, high] pair")
        return (low, high)

    def done(self) -> None:
        unknown = set(self._mapping) - self._consumed
        if unknown:
            raise ContractError(f"unknown keys under {self._path}: {sorted(unknown)}")


# ---------------------------------------------------------------------------
# Typed contract sections


@dataclass(frozen=True)
class ProtocolMeta:
    project: str
    version: str
    status: str
    primary_claim: str
    require_clean_worktree: bool
    freeze_parent_commit: str


@dataclass(frozen=True)
class UnresolvedDecision:
    decision_id: str
    gate: str
    resolution_mode: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class IntegrationGate:
    one_step_normalized_error_p99_max: float
    one_step_normalized_error_absolute_max: float
    eight_second_normalized_rmse_max: float
    eight_second_normalized_error_absolute_max: float
    terminal_event_classification_must_match: bool
    float32_vs_float64_one_step_normalized_error_max: float


@dataclass(frozen=True)
class Numerics:
    truth_dtype: str
    planning_dtype: str
    control_dt_s: float
    integrator: str
    substeps_per_control_step: int
    allowed_substep_candidates: tuple[int, ...]
    substep_selection_rule: str
    integration_gate: IntegrationGate
    reference_integrator: str
    reference_rtol: float
    reference_atol: float
    deterministic_algorithms: bool
    root_seed: int


@dataclass(frozen=True)
class FrictionSpec:
    viscous_nm_s_rad: tuple[float, float]
    coulomb_nm: tuple[float, float]
    low_speed_peak_nm: tuple[float, float]
    stribeck_velocity_rad_s: tuple[float, float]
    smoothing_velocity_rad_s: tuple[float, float]


@dataclass(frozen=True)
class ActuatorSpec:
    gain: tuple[float, float]
    deadzone_nm: tuple[float, float]


@dataclass(frozen=True)
class WorldSpec:
    world_id: str
    role: str
    components: tuple[str, ...]
    payload_kg: float | None = None
    friction: FrictionSpec | None = None
    actuator: ActuatorSpec | None = None
    elastic_coupling_nm: float | None = None
    base_world: str | None = None
    magnitude_scale: float | None = None


@dataclass(frozen=True)
class ScenarioGeneratorConfig:
    version: int
    bank_counts: dict[str, int]
    maximum_attempts_per_scenario: int
    joint_structural_strata: int
    initial_q1_rad: tuple[float, float]
    initial_q2_rad: tuple[float, float]
    initial_qd_rad_s: tuple[float, float]
    hard_joint_margin_rad: float
    minimum_signed_obstacle_clearance_m: float
    x_sectors_m: dict[str, tuple[float, float]]
    y_m: tuple[float, float]
    radial_distance_m: tuple[float, float]
    minimum_pairwise_separation_m: float
    minimum_initial_to_first_distance_m: float
    total_ordered_chord_length_m: tuple[float, float]
    obstacle_chord_fraction: tuple[float, float]
    obstacle_radius_m: tuple[float, float]
    obstacle_perpendicular_offset_in_radii: tuple[float, float]
    obstacle_center_x_m: tuple[float, float]
    obstacle_center_y_m: tuple[float, float]
    minimum_target_center_clearance_beyond_radius_m: float
    ik_hard_limit_margin_rad: float
    obstacle_inflation_beyond_arm_radius_m: float
    q1_grid_points: int
    q2_grid_points: int
    edge_resolution_bound_m: float


@dataclass(frozen=True)
class CostWeights:
    position: float
    near_target_velocity: float
    torque: float
    torque_change: float
    obstacle_barrier: float
    joint_barrier: float
    terminal_position: float
    remaining_target: float
    invalid_candidate: float
    obstacle_soft_margin_m: float
    joint_soft_margin_rad: float


@dataclass(frozen=True)
class TaskConfig:
    name: str
    target_count: int
    target_radius_m: float
    target_speed_threshold_m_s: float
    near_target_velocity_radius_m: float
    target_dwell_steps: int
    timeout_steps: int
    obstacle_shape: str
    arm_safety_radius_m: float
    swept_collision_max_workspace_step_m: float
    swept_collision_inflation_m: float
    scenario_generator: ScenarioGeneratorConfig
    cost: CostWeights


@dataclass(frozen=True)
class ExplorationConfig:
    band_limited_random_fraction: float
    multisine_fraction: float
    nominal_mpc_perturbed_fraction: float
    torque_envelope_nm: tuple[float, float]
    random_sample_rate_hz: float
    random_filter: str
    random_cutoff_hz: float
    random_burn_in_samples: int
    multisine_frequencies_hz: tuple[float, ...]
    mpc_perturbation_scenario_bank: str
    mpc_perturbation_low_nm: float
    mpc_perturbation_high_nm: float
    mpc_perturbation_hold_steps: int


@dataclass(frozen=True)
class ResetStrata:
    shoulder_bands_rad: tuple[tuple[float, float], ...]
    shoulder_weights: tuple[float, ...]
    elbow_bands_rad: tuple[tuple[float, float], ...]
    elbow_weights: tuple[float, ...]
    velocity_low_abs_max_rad_s: float
    velocity_moderate_abs_range_rad_s: tuple[float, float]
    velocity_weights: tuple[float, ...]


@dataclass(frozen=True)
class DataConfig:
    collection_unit_valid_transitions: int
    adaptation_budgets_total: tuple[int, ...]
    primary_budget_total: int
    train_fraction: float
    validation_fraction: float
    prediction_test_transitions: int
    split_unit: str
    budgets_are_nested: bool
    rollout_horizons: tuple[int, ...]
    exploration: ExplorationConfig
    reset_strata: ResetStrata
    model_inputs: tuple[str, ...]
    velocity_scale_rad_s: float
    acceleration_scale_rad_s2: tuple[float, float]
    empirical_normalizer_forbidden: bool


@dataclass(frozen=True)
class FittedPhysicsConfig:
    parameters: tuple[str, ...]
    bounds: dict[str, tuple[float, float]]
    deterministic_restarts: int


@dataclass(frozen=True)
class NeuralCommonConfig:
    input_dim: int
    hidden_widths: tuple[int, ...]
    activation: str
    output_dim: int
    output: str
    ensemble_members: int


@dataclass(frozen=True)
class ModelsConfig:
    required: tuple[str, ...]
    fitted_physics: FittedPhysicsConfig
    neural_common: NeuralCommonConfig
    residual_final_layer_zero_init: bool
    residual_penalty_primary: float
    nominal_sanity_residual_success_degradation_absolute_max: float
    nominal_sanity_residual_twenty_step_rmse_ratio_to_blackbox_max: float


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str
    updates: int
    allowed_update_candidates: tuple[int, ...]
    update_selection_relative_tolerance: float
    one_step_batch_size: int
    rollout_batch_size: int
    rollout_horizon: int
    learning_rate_initial: float
    learning_rate_final: float
    schedule: str
    weight_decay: float
    gradient_norm_clip: float
    validation_every_updates: int
    checkpoint_rule: str
    one_step_weight: float
    five_step_weight: float


@dataclass(frozen=True)
class ControllerProfile:
    candidates: int
    elites: int
    replan_every_steps: int
    execute_actions_per_plan: int


@dataclass(frozen=True)
class PlanningConfig:
    method: str
    profile: str
    allowed_profiles: dict[str, ControllerProfile]
    horizon_steps: int
    action_knots: int
    candidates: int
    elites: int
    iterations: int
    initial_latent_std: float
    latent_std_floor: float
    old_distribution_retention: float
    replan_every_steps: int
    execute_actions_per_plan: int
    exact_reference_worlds: tuple[str, ...]
    exact_reference_success_min_each_of_40: int
    nominal_composite_success_range: tuple[int, int]
    evaluation_ceiling_gpu_hours: float
    required_reserve_fraction: float


@dataclass(frozen=True)
class EvaluationConfig:
    scope_profile: str
    allowed_scope_profiles: tuple[str, ...]
    scope_preference_order: tuple[str, ...]
    pipeline_replicates_primary: int
    scenario_families_primary: int
    primary_world: str
    primary_budget: int
    primary_methods: tuple[str, ...]
    all_reference_methods: tuple[str, ...]
    required_control_budgets: tuple[int, ...]
    intermediate_control_budgets: tuple[int, ...]
    include_intermediate_control_budgets: bool
    nonprimary_budget_pipeline_replicates: int
    nonprimary_budget_scenario_families: int
    prediction_budgets: tuple[int, ...]
    prediction_pipeline_replicates: int
    additional_primary_budget_prediction_pipelines: int
    mechanism_worlds: tuple[str, ...]
    mechanism_pipeline_replicates: dict[str, int]
    mechanism_scenario_families: dict[str, int]
    transfer_worlds: tuple[str, ...]
    transfer_pipeline_replicates: dict[str, int]
    transfer_scenario_families: int
    maximum_attempts_total: int


@dataclass(frozen=True)
class ContrastSpec:
    contrast_id: str
    weights: dict[str, float]


@dataclass(frozen=True)
class AnalysisConfig:
    primary_outcome: str
    primary_contrast: ContrastSpec
    practical_effect_threshold_absolute: float
    secondary_confirmatory_contrasts: tuple[ContrastSpec, ...]
    bootstrap_replicates: int
    bootstrap_interval: str
    protected_scenarios_per_stratum: int
    support_applicable_methods: tuple[str, ...]
    support_threshold: float
    result_states: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentContract:
    source_path: str
    schema_version: int
    protocol: ProtocolMeta
    unresolved_decisions: tuple[UnresolvedDecision, ...]
    calibration_maximum_proposals_per_gate: int
    numerics: Numerics
    arm: ArmParameters
    worlds: dict[str, WorldSpec]
    learned_development_worlds: tuple[str, ...]
    protected_worlds: tuple[str, ...]
    elastic_candidates_nm: tuple[float, ...]
    task: TaskConfig
    data: DataConfig
    models: ModelsConfig
    training: TrainingConfig
    planning: PlanningConfig
    evaluation: EvaluationConfig
    analysis: AnalysisConfig
    media: dict[str, Any] = field(repr=False, default_factory=dict)
    release: dict[str, Any] = field(repr=False, default_factory=dict)


# ---------------------------------------------------------------------------
# Section parsers


def _parse_protocol(node: _Node) -> ProtocolMeta:
    meta = ProtocolMeta(
        project=node.str_("project"),
        version=node.str_("version"),
        status=node.str_("status"),
        primary_claim=node.str_("primary_claim"),
        require_clean_worktree=node.bool_("require_clean_worktree"),
        freeze_parent_commit=node.str_("freeze_parent_commit"),
    )
    node.done()
    if meta.status not in {"DRAFT", "FROZEN", "SUPERSEDED"}:
        raise ContractError(f"protocol.status must be DRAFT/FROZEN/SUPERSEDED, got {meta.status}")
    return meta


def _parse_decisions(raw: Any) -> tuple[UnresolvedDecision, ...]:
    if not isinstance(raw, list):
        raise ContractError("unresolved_decisions must be a list")
    decisions: list[UnresolvedDecision] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ContractError(f"unresolved_decisions[{index}] must be a mapping")
        node = _Node(entry, f"unresolved_decisions[{index}]")
        decision = UnresolvedDecision(
            decision_id=node.str_("id"),
            gate=node.str_("gate"),
            resolution_mode=node.str_("resolution_mode"),
            paths=node.str_list("paths"),
        )
        node.done()
        if decision.decision_id in seen:
            raise ContractError(f"duplicate decision id {decision.decision_id}")
        if decision.resolution_mode not in {
            "deterministic_selector",
            "method_blind_human_calibration",
        }:
            raise ContractError(
                f"unknown resolution mode {decision.resolution_mode} "
                f"for decision {decision.decision_id}"
            )
        seen.add(decision.decision_id)
        decisions.append(decision)
    return tuple(decisions)


def _parse_numerics(node: _Node) -> Numerics:
    gate_node = node.child("integration_gate")
    gate = IntegrationGate(
        one_step_normalized_error_p99_max=gate_node.float_(
            "one_step_normalized_error_p99_max", positive=True
        ),
        one_step_normalized_error_absolute_max=gate_node.float_(
            "one_step_normalized_error_absolute_max", positive=True
        ),
        eight_second_normalized_rmse_max=gate_node.float_(
            "eight_second_normalized_rmse_max", positive=True
        ),
        eight_second_normalized_error_absolute_max=gate_node.float_(
            "eight_second_normalized_error_absolute_max", positive=True
        ),
        terminal_event_classification_must_match=gate_node.bool_(
            "terminal_event_classification_must_match"
        ),
        float32_vs_float64_one_step_normalized_error_max=gate_node.float_(
            "float32_vs_float64_one_step_normalized_error_max", positive=True
        ),
    )
    gate_node.done()
    numerics = Numerics(
        truth_dtype=node.str_("truth_dtype"),
        planning_dtype=node.str_("planning_dtype_all_methods"),
        control_dt_s=node.float_("control_dt_s", positive=True),
        integrator=node.str_("integrator"),
        substeps_per_control_step=node.int_("substeps_per_control_step", minimum=1),
        allowed_substep_candidates=node.int_list("allowed_substep_candidates"),
        substep_selection_rule=node.str_("substep_selection_rule"),
        integration_gate=gate,
        reference_integrator=node.str_("reference_integrator"),
        reference_rtol=node.float_("reference_rtol", positive=True),
        reference_atol=node.float_("reference_atol", positive=True),
        deterministic_algorithms=node.bool_("deterministic_algorithms"),
        root_seed=node.int_("root_seed", minimum=0),
    )
    node.done()
    if numerics.substeps_per_control_step not in numerics.allowed_substep_candidates:
        raise ContractError("numerics.substeps_per_control_step not among allowed candidates")
    return numerics


def _parse_arm(node: _Node) -> ArmParameters:
    convention = node.child("coordinate_convention")
    convention.str_("q1")
    convention.str_("q2")
    convention.str_("gravity_direction")
    convention.done()
    arm = ArmParameters(
        link_lengths_m=node.float_pair("link_length_m", positive=True),
        com_lengths_m=node.float_pair("com_length_m", positive=True),
        masses_kg=node.float_pair("link_mass_kg", positive=True),
        inertias_kg_m2=node.float_pair("com_inertia_kg_m2", positive=True),
        viscous_nm_s_rad=node.float_pair("nominal_viscous_nm_s_rad"),
        gravity_m_s2=node.float_("gravity_m_s2", positive=True),
        torque_limit_nm=node.float_pair("torque_limit_nm", positive=True),
        q_min_rad=node.float_pair("q_min_rad"),
        q_max_rad=node.float_pair("q_max_rad"),
        speed_limit_rad_s=node.float_pair("speed_limit_rad_s", positive=True),
    )
    if not node.bool_("state_clipping_forbidden"):
        raise ContractError("arm.state_clipping_forbidden must be true")
    node.done()
    for joint in range(2):
        if arm.q_min_rad[joint] >= arm.q_max_rad[joint]:
            raise ContractError(f"arm joint {joint} has empty angle range")
        if arm.com_lengths_m[joint] > arm.link_lengths_m[joint]:
            raise ContractError(f"arm joint {joint} COM lies beyond the link")
    return arm


def _parse_friction(node: _Node) -> FrictionSpec:
    spec = FrictionSpec(
        viscous_nm_s_rad=node.float_pair("viscous_nm_s_rad"),
        coulomb_nm=node.float_pair("coulomb_nm"),
        low_speed_peak_nm=node.float_pair("low_speed_peak_nm"),
        stribeck_velocity_rad_s=node.float_pair("stribeck_velocity_rad_s", positive=True),
        smoothing_velocity_rad_s=node.float_pair("smoothing_velocity_rad_s", positive=True),
    )
    node.done()
    for joint in range(2):
        if spec.low_speed_peak_nm[joint] < spec.coulomb_nm[joint]:
            raise ContractError("friction low-speed peak must be >= Coulomb level")
    return spec


def _parse_actuator(node: _Node) -> ActuatorSpec:
    spec = ActuatorSpec(
        gain=node.float_pair("gain", positive=True),
        deadzone_nm=node.float_pair("deadzone_nm"),
    )
    node.done()
    if any(value < 0 for value in spec.deadzone_nm):
        raise ContractError("actuator dead zones must be non-negative")
    return spec


_KNOWN_COMPONENTS = {"payload", "nonlinear_friction", "actuator", "elastic_coupling"}


def _parse_world(world_id: str, node: _Node) -> WorldSpec:
    role = node.str_("role")
    components: tuple[str, ...] = ()
    payload: float | None = None
    friction: FrictionSpec | None = None
    actuator: ActuatorSpec | None = None
    elastic: float | None = None
    base_world: str | None = None
    scale: float | None = None
    if node.has("base_world"):
        base_world = node.str_("base_world")
        if node.has("magnitude_scale"):
            scale = node.float_("magnitude_scale", positive=True)
        if node.has("additional_components"):
            components = node.str_list("additional_components")
        if node.has("elastic_coupling_nm"):
            elastic = node.float_("elastic_coupling_nm", positive=True)
    else:
        components = node.str_list("components")
        if node.has("payload_kg"):
            payload = node.float_("payload_kg", minimum=0.0)
        if node.has("friction"):
            friction = _parse_friction(node.child("friction"))
        if node.has("actuator"):
            actuator = _parse_actuator(node.child("actuator"))
    node.done()
    unknown = set(components) - _KNOWN_COMPONENTS
    if unknown:
        raise ContractError(f"world {world_id} has unknown components {sorted(unknown)}")
    if "payload" in components and payload is None:
        raise ContractError(f"world {world_id} enables payload without payload_kg")
    if "nonlinear_friction" in components and friction is None:
        raise ContractError(f"world {world_id} enables friction without parameters")
    if "actuator" in components and actuator is None:
        raise ContractError(f"world {world_id} enables actuator without parameters")
    if "elastic_coupling" in components and elastic is None:
        raise ContractError(f"world {world_id} enables elastic coupling without magnitude")
    return WorldSpec(
        world_id=world_id,
        role=role,
        components=components,
        payload_kg=payload,
        friction=friction,
        actuator=actuator,
        elastic_coupling_nm=elastic,
        base_world=base_world,
        magnitude_scale=scale,
    )


def _parse_scenario_generator(node: _Node) -> ScenarioGeneratorConfig:
    banks_node = node.child("banks")
    bank_counts: dict[str, int] = {}
    for bank in ("calibration", "pilot", "training_task", "protected"):
        bank_counts[bank] = banks_node.int_(bank, minimum=1)
    banks_node.done()
    initial = node.child("initial_state")
    targets = node.child("target_sampling")
    sectors = targets.child("x_sectors_m")
    x_sectors = {
        "left": sectors.float_range("left"),
        "middle": sectors.float_range("middle"),
        "right": sectors.float_range("right"),
    }
    sectors.done()
    obstacle = node.child("obstacle_sampling")
    feasibility = node.child("structural_feasibility")
    config = ScenarioGeneratorConfig(
        version=node.int_("version", minimum=1),
        bank_counts=bank_counts,
        maximum_attempts_per_scenario=node.int_("maximum_attempts_per_scenario", minimum=1),
        joint_structural_strata=node.int_("joint_structural_strata", minimum=1),
        initial_q1_rad=initial.float_range("q1_rad"),
        initial_q2_rad=initial.float_range("q2_rad"),
        initial_qd_rad_s=initial.float_range("qd_each_rad_s"),
        hard_joint_margin_rad=initial.float_("hard_joint_margin_rad", positive=True),
        minimum_signed_obstacle_clearance_m=initial.float_(
            "minimum_signed_obstacle_clearance_m", positive=True
        ),
        x_sectors_m=x_sectors,
        y_m=targets.float_range("y_m"),
        radial_distance_m=targets.float_range("radial_distance_m"),
        minimum_pairwise_separation_m=targets.float_(
            "minimum_pairwise_separation_m", positive=True
        ),
        minimum_initial_to_first_distance_m=targets.float_(
            "minimum_initial_to_first_distance_m", positive=True
        ),
        total_ordered_chord_length_m=targets.float_range("total_ordered_chord_length_m"),
        obstacle_chord_fraction=obstacle.float_range("chord_fraction"),
        obstacle_radius_m=obstacle.float_range("radius_m"),
        obstacle_perpendicular_offset_in_radii=obstacle.float_range(
            "perpendicular_offset_in_obstacle_radii"
        ),
        obstacle_center_x_m=obstacle.float_range("center_x_m"),
        obstacle_center_y_m=obstacle.float_range("center_y_m"),
        minimum_target_center_clearance_beyond_radius_m=obstacle.float_(
            "minimum_target_center_clearance_beyond_radius_m", positive=True
        ),
        ik_hard_limit_margin_rad=feasibility.float_("ik_hard_limit_margin_rad", positive=True),
        obstacle_inflation_beyond_arm_radius_m=feasibility.float_(
            "obstacle_inflation_beyond_arm_radius_m", positive=True
        ),
        q1_grid_points=feasibility.int_("q1_grid_points", minimum=2),
        q2_grid_points=feasibility.int_("q2_grid_points", minimum=2),
        edge_resolution_bound_m=feasibility.float_("edge_resolution_bound_m", positive=True),
    )
    initial.done()
    targets.done()
    obstacle.done()
    feasibility.done()
    node.done()
    return config


def _parse_task(node: _Node) -> TaskConfig:
    cost_node = node.child("cost")
    cost = CostWeights(
        position=cost_node.float_("position", minimum=0.0),
        near_target_velocity=cost_node.float_("near_target_velocity", minimum=0.0),
        torque=cost_node.float_("torque", minimum=0.0),
        torque_change=cost_node.float_("torque_change", minimum=0.0),
        obstacle_barrier=cost_node.float_("obstacle_barrier", minimum=0.0),
        joint_barrier=cost_node.float_("joint_barrier", minimum=0.0),
        terminal_position=cost_node.float_("terminal_position", minimum=0.0),
        remaining_target=cost_node.float_("remaining_target", minimum=0.0),
        invalid_candidate=cost_node.float_("invalid_candidate", positive=True),
        obstacle_soft_margin_m=cost_node.float_("obstacle_soft_margin_m", positive=True),
        joint_soft_margin_rad=cost_node.float_("joint_soft_margin_rad", positive=True),
    )
    cost_node.done()
    task = TaskConfig(
        name=node.str_("name"),
        target_count=node.int_("target_count", minimum=1),
        target_radius_m=node.float_("target_radius_m", positive=True),
        target_speed_threshold_m_s=node.float_("target_speed_threshold_m_s", positive=True),
        near_target_velocity_radius_m=node.float_(
            "near_target_velocity_radius_m", positive=True
        ),
        target_dwell_steps=node.int_("target_dwell_steps", minimum=1),
        timeout_steps=node.int_("timeout_steps", minimum=1),
        obstacle_shape=node.str_("obstacle_shape"),
        arm_safety_radius_m=node.float_("arm_safety_radius_m", positive=True),
        swept_collision_max_workspace_step_m=node.float_(
            "swept_collision_max_workspace_step_m", positive=True
        ),
        swept_collision_inflation_m=node.float_("swept_collision_inflation_m", positive=True),
        scenario_generator=_parse_scenario_generator(node.child("scenario_generator")),
        cost=cost,
    )
    node.done()
    if task.near_target_velocity_radius_m < task.target_radius_m:
        raise ContractError("near-target velocity radius must not be smaller than target radius")
    return task


def _parse_data(node: _Node) -> DataConfig:
    exploration_node = node.child("exploration")
    random_node = exploration_node.child("band_limited_random")
    multisine_node = exploration_node.child("multisine")
    perturbation_node = exploration_node.child("nominal_mpc_perturbation")
    exploration = ExplorationConfig(
        band_limited_random_fraction=exploration_node.float_(
            "band_limited_random_fraction", minimum=0.0
        ),
        multisine_fraction=exploration_node.float_("multisine_fraction", minimum=0.0),
        nominal_mpc_perturbed_fraction=exploration_node.float_(
            "nominal_mpc_perturbed_fraction", minimum=0.0
        ),
        torque_envelope_nm=exploration_node.float_pair("torque_envelope_nm", positive=True),
        random_sample_rate_hz=random_node.float_("sample_rate_hz", positive=True),
        random_filter=random_node.str_("filter"),
        random_cutoff_hz=random_node.float_("cutoff_hz", positive=True),
        random_burn_in_samples=random_node.int_("burn_in_samples", minimum=0),
        multisine_frequencies_hz=multisine_node.float_list("frequencies_hz"),
        mpc_perturbation_scenario_bank=perturbation_node.str_("scenario_bank"),
        mpc_perturbation_low_nm=perturbation_node.float_("low_nm"),
        mpc_perturbation_high_nm=perturbation_node.float_("high_nm"),
        mpc_perturbation_hold_steps=perturbation_node.int_("hold_steps", minimum=1),
    )
    random_node.done()
    multisine_node.done()
    perturbation_node.done()
    exploration_node.done()

    strata_node = node.child("reset_strata")
    shoulder_node = strata_node.child("shoulder_rad")
    elbow_node = strata_node.child("elbow_rad")
    velocity_node = strata_node.child("velocity_regimes")

    def _bands(bands_node: _Node) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
        raw_bands = bands_node.raw("bands")
        if not isinstance(raw_bands, list):
            raise ContractError("reset strata bands must be a list of [low, high] pairs")
        bands: list[tuple[float, float]] = []
        for pair in raw_bands:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(v, (int, float)) for v in pair)
            ):
                raise ContractError("each reset band must be a [low, high] number pair")
            low, high = float(pair[0]), float(pair[1])
            if low >= high:
                raise ContractError("reset band must be increasing")
            bands.append((low, high))
        weights = bands_node.float_list("weights")
        if len(weights) != len(bands):
            raise ContractError("reset band weights must match the number of bands")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ContractError("reset band weights must sum to one")
        return tuple(bands), weights

    shoulder_bands, shoulder_weights = _bands(shoulder_node)
    elbow_bands, elbow_weights = _bands(elbow_node)
    velocity_weights = velocity_node.float_list("weights")
    if abs(sum(velocity_weights) - 1.0) > 1e-9:
        raise ContractError("velocity regime weights must sum to one")
    strata = ResetStrata(
        shoulder_bands_rad=shoulder_bands,
        shoulder_weights=shoulder_weights,
        elbow_bands_rad=elbow_bands,
        elbow_weights=elbow_weights,
        velocity_low_abs_max_rad_s=velocity_node.float_("low_abs_max_rad_s", positive=True),
        velocity_moderate_abs_range_rad_s=velocity_node.float_range("moderate_abs_range_rad_s"),
        velocity_weights=velocity_weights,
    )
    shoulder_node.done()
    elbow_node.done()
    velocity_node.done()
    strata_node.done()

    data = DataConfig(
        collection_unit_valid_transitions=node.int_(
            "collection_unit_valid_transitions", minimum=1
        ),
        adaptation_budgets_total=node.int_list("adaptation_budgets_total"),
        primary_budget_total=node.int_("primary_budget_total", minimum=1),
        train_fraction=node.float_("train_fraction", positive=True),
        validation_fraction=node.float_("validation_fraction", positive=True),
        prediction_test_transitions=node.int_("prediction_test_transitions", minimum=1),
        split_unit=node.str_("split_unit"),
        budgets_are_nested=node.bool_("budgets_are_nested"),
        rollout_horizons=node.int_list("rollout_horizons"),
        exploration=exploration,
        reset_strata=strata,
        model_inputs=node.str_list("model_inputs"),
        velocity_scale_rad_s=node.float_("velocity_scale_rad_s", positive=True),
        acceleration_scale_rad_s2=node.float_pair("acceleration_scale_rad_s2", positive=True),
        empirical_normalizer_forbidden=node.bool_("empirical_normalizer_forbidden"),
    )
    node.done()

    fractions = (
        exploration.band_limited_random_fraction
        + exploration.multisine_fraction
        + exploration.nominal_mpc_perturbed_fraction
    )
    if abs(fractions - 1.0) > 1e-9:
        raise ContractError("exploration fractions must sum to one")
    if abs(data.train_fraction + data.validation_fraction - 1.0) > 1e-9:
        raise ContractError("train and validation fractions must sum to one")
    budgets = data.adaptation_budgets_total
    if list(budgets) != sorted(set(budgets)):
        raise ContractError("adaptation budgets must be strictly increasing and unique")
    if data.primary_budget_total not in budgets:
        raise ContractError("primary budget must be one of the adaptation budgets")
    unit = data.collection_unit_valid_transitions
    # Budgets split into whole collection units at an exact 3:1 ratio,
    # so every budget must be a multiple of four units.
    for budget in budgets:
        if budget % (4 * unit) != 0:
            raise ContractError(
                f"budget {budget} is not a multiple of four collection units ({4 * unit})"
            )
    if data.prediction_test_transitions % unit != 0:
        raise ContractError("prediction test size must be a whole number of collection units")
    if len(data.model_inputs) != 8:
        raise ContractError("model input feature list must have exactly eight entries")
    if not data.empirical_normalizer_forbidden:
        raise ContractError("empirical normalizers are forbidden in this study")
    return data


def _parse_models(node: _Node) -> ModelsConfig:
    fitted_node = node.child("fitted_physics")
    bounds_node = fitted_node.child("bounds")
    parameters = fitted_node.str_list("parameters")
    bounds: dict[str, tuple[float, float]] = {}
    for name in parameters:
        bounds[name] = bounds_node.float_range(name)
    bounds_node.done()
    fitted = FittedPhysicsConfig(
        parameters=parameters,
        bounds=bounds,
        deterministic_restarts=fitted_node.int_("deterministic_restarts", minimum=1),
    )
    fitted_node.done()

    neural_node = node.child("neural_common")
    neural = NeuralCommonConfig(
        input_dim=neural_node.int_("input_dim", minimum=1),
        hidden_widths=tuple(neural_node.int_list("hidden_widths")),
        activation=neural_node.str_("activation"),
        output_dim=neural_node.int_("output_dim", minimum=1),
        output=neural_node.str_("output"),
        ensemble_members=neural_node.int_("ensemble_members", minimum=1),
    )
    neural_node.done()

    residual_node = node.child("residual")
    zero_init = residual_node.bool_("final_layer_zero_init")
    residual_penalty = residual_node.float_("residual_penalty_primary", minimum=0.0)
    residual_node.done()

    sanity_node = node.child("sanity_tolerances")
    models = ModelsConfig(
        required=node.str_list("required"),
        fitted_physics=fitted,
        neural_common=neural,
        residual_final_layer_zero_init=zero_init,
        residual_penalty_primary=residual_penalty,
        nominal_sanity_residual_success_degradation_absolute_max=sanity_node.float_(
            "nominal_sanity_residual_success_degradation_absolute_max", minimum=0.0
        ),
        nominal_sanity_residual_twenty_step_rmse_ratio_to_blackbox_max=sanity_node.float_(
            "nominal_sanity_residual_twenty_step_rmse_ratio_to_blackbox_max", positive=True
        ),
    )
    sanity_node.done()
    node.done()
    if not zero_init:
        raise ContractError("the residual head must be zero-initialized (that is the prior)")
    if models.neural_common.output != "joint_acceleration":
        raise ContractError("both neural methods must predict joint acceleration")
    return models


def _parse_training(node: _Node) -> TrainingConfig:
    training = TrainingConfig(
        optimizer=node.str_("optimizer"),
        updates=node.int_("updates", minimum=1),
        allowed_update_candidates=node.int_list("allowed_update_candidates"),
        update_selection_relative_tolerance=node.float_(
            "update_selection_relative_tolerance", positive=True
        ),
        one_step_batch_size=node.int_("one_step_batch_size", minimum=1),
        rollout_batch_size=node.int_("rollout_batch_size", minimum=1),
        rollout_horizon=node.int_("rollout_horizon", minimum=1),
        learning_rate_initial=node.float_("learning_rate_initial", positive=True),
        learning_rate_final=node.float_("learning_rate_final", positive=True),
        schedule=node.str_("schedule"),
        weight_decay=node.float_("weight_decay", minimum=0.0),
        gradient_norm_clip=node.float_("gradient_norm_clip", positive=True),
        validation_every_updates=node.int_("validation_every_updates", minimum=1),
        checkpoint_rule=node.str_("checkpoint_rule"),
        one_step_weight=node.float_("one_step_weight", minimum=0.0),
        five_step_weight=node.float_("five_step_weight", minimum=0.0),
    )
    node.done()
    if training.updates not in training.allowed_update_candidates:
        raise ContractError("training.updates must be one of the allowed candidates")
    return training


def _parse_planning(node: _Node) -> PlanningConfig:
    profiles_node = node.child("allowed_profiles")
    profiles: dict[str, ControllerProfile] = {}
    for profile_id in ("A", "B", "C"):
        profile_node = profiles_node.child(profile_id)
        profiles[profile_id] = ControllerProfile(
            candidates=profile_node.int_("candidates", minimum=2),
            elites=profile_node.int_("elites", minimum=1),
            replan_every_steps=profile_node.int_("replan_every_steps", minimum=1),
            execute_actions_per_plan=profile_node.int_("execute_actions_per_plan", minimum=1),
        )
        profile_node.done()
    profiles_node.done()

    calibration_node = node.child("g2_calibration")
    nominal_range = calibration_node.int_list("nominal_composite_success_range_inclusive_of_40")
    if len(nominal_range) != 2 or nominal_range[0] > nominal_range[1]:
        raise ContractError("nominal composite success range must be an ordered pair")
    planning = PlanningConfig(
        method=node.str_("method"),
        profile=node.str_("profile"),
        allowed_profiles=profiles,
        horizon_steps=node.int_("horizon_steps", minimum=1),
        action_knots=node.int_("action_knots", minimum=1),
        candidates=node.int_("candidates", minimum=2),
        elites=node.int_("elites", minimum=1),
        iterations=node.int_("iterations", minimum=1),
        initial_latent_std=node.float_("initial_latent_std", positive=True),
        latent_std_floor=node.float_("latent_std_floor", positive=True),
        old_distribution_retention=node.float_("old_distribution_retention", minimum=0.0),
        replan_every_steps=node.int_("replan_every_steps", minimum=1),
        execute_actions_per_plan=node.int_("execute_actions_per_plan", minimum=1),
        exact_reference_worlds=calibration_node.str_list("exact_reference_worlds"),
        exact_reference_success_min_each_of_40=calibration_node.int_(
            "exact_reference_success_min_each_of_40", minimum=0
        ),
        nominal_composite_success_range=(nominal_range[0], nominal_range[1]),
        evaluation_ceiling_gpu_hours=calibration_node.float_(
            "evaluation_ceiling_gpu_hours", positive=True
        ),
        required_reserve_fraction=calibration_node.float_(
            "required_reserve_fraction", minimum=0.0
        ),
    )
    calibration_node.done()
    node.done()
    selected = planning.allowed_profiles.get(planning.profile)
    if selected is None:
        raise ContractError(f"planning.profile {planning.profile} is not an allowed profile")
    if (
        planning.candidates != selected.candidates
        or planning.elites != selected.elites
        or planning.replan_every_steps != selected.replan_every_steps
        or planning.execute_actions_per_plan != selected.execute_actions_per_plan
    ):
        raise ContractError(
            "planning.{candidates,elites,replan_every_steps,execute_actions_per_plan} "
            "must equal the selected profile's values"
        )
    if planning.elites >= planning.candidates:
        raise ContractError("planning must have more candidates than elites")
    if planning.horizon_steps % planning.action_knots != 0:
        raise ContractError("horizon must be a whole multiple of the action knot count")
    return planning


def _parse_evaluation(node: _Node) -> EvaluationConfig:
    def _profile_ints(key: str) -> dict[str, int]:
        sub = node.child(key)
        result = {
            "balanced": sub.int_("balanced", minimum=0),
            "expanded": sub.int_("expanded", minimum=0),
        }
        sub.done()
        return result

    evaluation = EvaluationConfig(
        scope_profile=node.str_("scope_profile"),
        allowed_scope_profiles=node.str_list("allowed_scope_profiles"),
        scope_preference_order=node.str_list("scope_preference_order"),
        pipeline_replicates_primary=node.int_("pipeline_replicates_primary", minimum=1),
        scenario_families_primary=node.int_("scenario_families_primary", minimum=1),
        primary_world=node.str_("primary_world"),
        primary_budget=node.int_("primary_budget", minimum=1),
        primary_methods=node.str_list("primary_methods"),
        all_reference_methods=node.str_list("all_reference_methods"),
        required_control_budgets=node.int_list("required_control_budgets"),
        intermediate_control_budgets=node.int_list("intermediate_control_budgets"),
        include_intermediate_control_budgets=node.bool_(
            "include_intermediate_control_budgets"
        ),
        nonprimary_budget_pipeline_replicates=node.int_(
            "nonprimary_budget_pipeline_replicates", minimum=1
        ),
        nonprimary_budget_scenario_families=node.int_(
            "nonprimary_budget_scenario_families", minimum=1
        ),
        prediction_budgets=node.int_list("prediction_budgets"),
        prediction_pipeline_replicates=node.int_("prediction_pipeline_replicates", minimum=1),
        additional_primary_budget_prediction_pipelines=node.int_(
            "additional_primary_budget_prediction_pipelines", minimum=0
        ),
        mechanism_worlds=node.str_list("mechanism_worlds"),
        mechanism_pipeline_replicates=_profile_ints("mechanism_pipeline_replicates"),
        mechanism_scenario_families=_profile_ints("mechanism_scenario_families"),
        transfer_worlds=node.str_list("transfer_worlds"),
        transfer_pipeline_replicates=_profile_ints("transfer_pipeline_replicates"),
        transfer_scenario_families=node.int_("transfer_scenario_families", minimum=1),
        maximum_attempts_total=node.int_("maximum_attempts_total", minimum=1),
    )
    node.done()
    if evaluation.scope_profile not in evaluation.allowed_scope_profiles:
        raise ContractError("evaluation.scope_profile must be an allowed profile")
    if sorted(evaluation.scope_preference_order) != sorted(evaluation.allowed_scope_profiles):
        raise ContractError("scope preference order must permute the allowed profiles")
    if evaluation.primary_budget not in evaluation.required_control_budgets:
        raise ContractError("primary budget must be a required control budget")
    if set(evaluation.primary_methods) != {"residual", "blackbox"}:
        raise ContractError("the primary contrast compares residual and blackbox")
    return evaluation


def _parse_contrast(entry: Any, path: str) -> ContrastSpec:
    if not isinstance(entry, dict):
        raise ContractError(f"{path} must be a mapping")
    node = _Node(entry, path)
    contrast_id = node.str_("id")
    weights_raw = node.raw("weights")
    node.done()
    if not isinstance(weights_raw, dict):
        raise ContractError(f"{path}.weights must be a mapping")
    weights: dict[str, float] = {}
    for method, weight in weights_raw.items():
        if not isinstance(method, str) or not isinstance(weight, (int, float)):
            raise ContractError(f"{path}.weights entries must map method -> number")
        weights[method] = float(weight)
    if abs(sum(weights.values())) > 1e-12:
        raise ContractError(f"{path} weights must sum to zero")
    return ContrastSpec(contrast_id=contrast_id, weights=weights)


def _parse_analysis(node: _Node) -> AnalysisConfig:
    bootstrap_node = node.child("bootstrap")
    support_node = node.child("support_metric")
    secondary_raw = node.raw("secondary_confirmatory_contrasts")
    if not isinstance(secondary_raw, list):
        raise ContractError("secondary_confirmatory_contrasts must be a list")
    analysis = AnalysisConfig(
        primary_outcome=node.str_("primary_outcome"),
        primary_contrast=_parse_contrast(node.raw("primary_contrast"), "analysis.primary_contrast"),
        practical_effect_threshold_absolute=node.float_(
            "practical_effect_threshold_absolute", positive=True
        ),
        secondary_confirmatory_contrasts=tuple(
            _parse_contrast(entry, f"analysis.secondary_confirmatory_contrasts[{index}]")
            for index, entry in enumerate(secondary_raw)
        ),
        bootstrap_replicates=bootstrap_node.int_("replicates", minimum=1),
        bootstrap_interval=bootstrap_node.str_("interval"),
        protected_scenarios_per_stratum=bootstrap_node.int_(
            "protected_scenarios_per_stratum", minimum=1
        ),
        support_applicable_methods=support_node.str_list("applicable_methods"),
        support_threshold=support_node.float_("threshold", positive=True),
        result_states=node.str_list("result_states"),
    )
    bootstrap_node.done()
    support_node.done()
    node.done()
    return analysis


# ---------------------------------------------------------------------------
# Whole-contract assembly and cross-checks


def _cross_check(contract: ExperimentContract) -> None:
    worlds = contract.worlds
    primary_worlds = [w for w in worlds.values() if w.role == "primary"]
    if len(primary_worlds) != 1:
        raise ContractError("exactly one world must have role: primary")
    if contract.evaluation.primary_world != primary_worlds[0].world_id:
        raise ContractError("evaluation.primary_world must be the world with role: primary")

    for world in worlds.values():
        if world.base_world is not None and world.base_world not in worlds:
            raise ContractError(
                f"world {world.world_id} derives from unknown base {world.base_world}"
            )

    # The standard mechanism components must equal their composite copies.
    composite = worlds.get(contract.evaluation.primary_world)
    assert composite is not None
    for mechanism_id, attribute in (
        ("payload_standard", "payload_kg"),
        ("friction_standard", "friction"),
        ("actuator_standard", "actuator"),
    ):
        mechanism = worlds.get(mechanism_id)
        if mechanism is None:
            continue
        if getattr(mechanism, attribute) != getattr(composite, attribute):
            raise ContractError(
                f"{mechanism_id}.{attribute} must equal its copy inside "
                f"{composite.world_id} (single source of mismatch truth)"
            )

    referenced = (
        set(contract.evaluation.mechanism_worlds)
        | set(contract.evaluation.transfer_worlds)
        | {contract.evaluation.primary_world}
        | set(contract.learned_development_worlds)
        | set(contract.protected_worlds)
        | set(contract.planning.exact_reference_worlds)
    )
    unknown_worlds = referenced - set(worlds)
    if unknown_worlds:
        raise ContractError(f"referenced worlds do not exist: {sorted(unknown_worlds)}")

    known_methods = set(contract.models.required)
    for contrast in (
        contract.analysis.primary_contrast,
        *contract.analysis.secondary_confirmatory_contrasts,
    ):
        unknown_methods = set(contrast.weights) - known_methods
        if unknown_methods:
            raise ContractError(
                f"contrast {contrast.contrast_id} references unknown methods "
                f"{sorted(unknown_methods)}"
            )

    if contract.evaluation.primary_budget != contract.data.primary_budget_total:
        raise ContractError("evaluation and data sections disagree on the primary budget")
    if set(contract.evaluation.required_control_budgets) - set(
        contract.data.adaptation_budgets_total
    ):
        raise ContractError("required control budgets must be adaptation budgets")
    if contract.training.rollout_horizon not in contract.data.rollout_horizons:
        raise ContractError("training rollout horizon must be a declared rollout horizon")
    if contract.models.neural_common.input_dim != len(contract.data.model_inputs):
        raise ContractError("neural input dimension must match the model input feature list")


def load_contract(path: Path) -> ExperimentContract:
    document = load_strict_yaml(path)
    root = _Node(document, "contract")
    schema_version = root.int_("schema_version", minimum=1)
    protocol = _parse_protocol(root.child("protocol"))
    decisions = _parse_decisions(root.raw("unresolved_decisions"))
    calibration_node = root.child("calibration_rules")
    maximum_proposals = calibration_node.int_("maximum_proposals_per_gate", minimum=1)
    calibration_node.str_list("forbidden_evidence")
    calibration_node.done()
    numerics = _parse_numerics(root.child("numerics"))
    arm = _parse_arm(root.child("arm"))

    worlds_node = root.raw("worlds")
    if not isinstance(worlds_node, dict):
        raise ContractError("worlds must be a mapping")
    worlds: dict[str, WorldSpec] = {}
    for world_id, world_raw in worlds_node.items():
        if not isinstance(world_id, str) or not isinstance(world_raw, dict):
            raise ContractError("worlds must map world id -> mapping")
        worlds[world_id] = _parse_world(world_id, _Node(world_raw, f"worlds.{world_id}"))

    access_node = root.child("world_access")
    learned_development = access_node.str_list("learned_development_worlds_before_G9")
    protected = access_node.str_list("protected_until_G9")
    access_node.str_list("permitted_pre_G9_checks_on_protected_worlds")
    elastic_node = access_node.child("elastic_selection")
    elastic_candidates = elastic_node.float_list("candidates_nm")
    elastic_node.str_("candidate_order")
    elastic_node.int_("exact_reference_success_min_of_40", minimum=0)
    elastic_node.str_("selection_rule")
    elastic_node.done()
    access_node.done()

    contract = ExperimentContract(
        source_path=str(path),
        schema_version=schema_version,
        protocol=protocol,
        unresolved_decisions=decisions,
        calibration_maximum_proposals_per_gate=maximum_proposals,
        numerics=numerics,
        arm=arm,
        worlds=worlds,
        learned_development_worlds=learned_development,
        protected_worlds=protected,
        elastic_candidates_nm=elastic_candidates,
        task=_parse_task(root.child("task")),
        data=_parse_data(root.child("data")),
        models=_parse_models(root.child("models")),
        training=_parse_training(root.child("training")),
        planning=_parse_planning(root.child("planning")),
        evaluation=_parse_evaluation(root.child("evaluation")),
        analysis=_parse_analysis(root.child("analysis")),
        media=dict(root.raw("media")),
        release=dict(root.raw("release")),
    )
    root.done()
    _cross_check(contract)
    return contract
