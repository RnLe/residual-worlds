"""Expansion of the evaluation scope into explicit, ordered job rows.

Nothing at evaluation time is discovered by globbing: this module
expands the contract's scope profile into every dataset, prediction-set,
fit, train, condition, prediction, and control job, each with a
precomputable identity. Panel membership is assigned here, before any
outcome exists:

* ``primary_RxS``: the full primary-cell crossing (all pipelines, all
  protected scenarios, all five methods at the primary budget);
* ``budget_curve``: the invariant subset -- the pipelines with the
  smallest pipeline-descriptor hashes and the smallest-hash scenario
  within each structural stratum -- used for every point joined into a
  budget curve. The reused primary-budget rows carry both memberships.

Composite-world control rows draw CEM noise from the method-, budget-,
and world-free ``composite_family`` namespace so the budget and
transfer contrasts share primitive randomness; all other rows use the
world/budget-qualified ``paired_methods`` namespace.

Execution order is a seeded block randomization over (world, budget,
pipeline, scenario) with methods rotated inside each block.
"""

from __future__ import annotations

from dataclasses import dataclass

from residual_worlds.config import ExperimentContract
from residual_worlds.identity import content_id
from residual_worlds.seeds import numpy_generator, seed_record
from residual_worlds.types import Scenario

ADAPTED_METHODS = ("fitted_physics", "blackbox", "residual")


@dataclass(frozen=True)
class ControlJob:
    job_id: str
    method: str
    world_id: str
    budget: int
    replicate: int
    scenario_id: str
    panel_memberships: tuple[str, ...]
    noise_family: str  # "composite_family" | "paired_methods"


@dataclass(frozen=True)
class PredictionJob:
    job_id: str
    method: str
    world_id: str
    budget: int
    replicate: int


@dataclass(frozen=True)
class TrainJob:
    method: str
    world_id: str
    budget: int
    replicate: int
    member: int


@dataclass(frozen=True)
class ExecutionManifest:
    dataset_jobs: tuple[tuple[str, int], ...]  # (world, replicate)
    prediction_set_jobs: tuple[tuple[str, int], ...]
    train_jobs: tuple[TrainJob, ...]
    prediction_jobs: tuple[PredictionJob, ...]
    control_jobs: tuple[ControlJob, ...]


def pipeline_descriptor_id(contract: ExperimentContract, replicate: int) -> str:
    """Outcome-independent pipeline identity (hash orders subset choices)."""
    return content_id(
        "spec",
        {
            "kind": "pipeline_descriptor",
            "replicate": replicate,
            "root_seed": contract.numerics.root_seed,
            "data_seed": seed_record(
                contract.numerics.root_seed, "data",
                contract.evaluation.primary_world, replicate,
            ).digest_sha256,
        },
    )


def subset_pipelines(contract: ExperimentContract, count: int) -> tuple[int, ...]:
    """The ``count`` primary-cell pipelines with smallest descriptor hashes."""
    replicates = range(contract.evaluation.pipeline_replicates_primary)
    ordered = sorted(replicates, key=lambda r: pipeline_descriptor_id(contract, r))
    return tuple(sorted(ordered[:count]))


def subset_scenarios(
    contract: ExperimentContract, scenarios: list[Scenario], count_per_stratum: int = 1
) -> tuple[str, ...]:
    """Smallest scenario-id hash within each structural stratum (ordered)."""
    by_stratum: dict[int, list[Scenario]] = {}
    for scenario in scenarios:
        by_stratum.setdefault(scenario.stratum_id, []).append(scenario)
    selected: list[str] = []
    for stratum in sorted(by_stratum):
        chosen = sorted(by_stratum[stratum], key=lambda s: s.scenario_id)
        selected.extend(s.scenario_id for s in chosen[:count_per_stratum])
    return tuple(selected)


def _control_job(
    contract: ExperimentContract,
    method: str,
    world_id: str,
    budget: int,
    replicate: int,
    scenario_id: str,
    panels: tuple[str, ...],
) -> ControlJob:
    family = (
        "composite_family"
        if world_id == contract.evaluation.primary_world
        or world_id in contract.evaluation.transfer_worlds
        else "paired_methods"
    )
    job_id = content_id(
        "evaluation_job",
        {
            "schema": 1,
            "method": method,
            "world_id": world_id,
            "budget": budget,
            "replicate": replicate,
            "scenario_id": scenario_id,
            "root_seed": contract.numerics.root_seed,
        },
    )
    return ControlJob(
        job_id=job_id,
        method=method,
        world_id=world_id,
        budget=budget,
        replicate=replicate,
        scenario_id=scenario_id,
        panel_memberships=panels,
        noise_family=family,
    )


def build_execution_manifest(
    contract: ExperimentContract, protected_scenarios: list[Scenario]
) -> ExecutionManifest:
    evaluation = contract.evaluation
    primary_world = evaluation.primary_world
    primary_budget = evaluation.primary_budget
    primary_replicates = tuple(range(evaluation.pipeline_replicates_primary))
    curve_pipelines = subset_pipelines(
        contract, evaluation.nonprimary_budget_pipeline_replicates
    )
    curve_scenarios = subset_scenarios(contract, protected_scenarios)
    if len(curve_scenarios) > evaluation.nonprimary_budget_scenario_families:
        curve_scenarios = curve_scenarios[: evaluation.nonprimary_budget_scenario_families]
    all_scenarios = tuple(s.scenario_id for s in protected_scenarios)[
        : evaluation.scenario_families_primary
    ]
    members = contract.models.neural_common.ensemble_members

    # Datasets and training: composite world at every prediction budget.
    train_jobs: list[TrainJob] = []
    prediction_jobs: list[PredictionJob] = []
    needed_replicates: set[int] = set()
    for budget in evaluation.prediction_budgets:
        if budget == primary_budget:
            replicates = primary_replicates
        else:
            replicates = curve_pipelines
        prediction_replicates = (
            tuple(
                sorted(
                    set(curve_pipelines)
                    | set(
                        subset_pipelines(
                            contract,
                            evaluation.nonprimary_budget_pipeline_replicates
                            + evaluation.additional_primary_budget_prediction_pipelines,
                        )
                    )
                )
            )
            if budget == primary_budget
            else curve_pipelines
        )
        needed_replicates.update(replicates)
        needed_replicates.update(prediction_replicates)
        for replicate in replicates:
            for method in ("blackbox", "residual"):
                for member in range(members):
                    train_jobs.append(
                        TrainJob(method, primary_world, budget, replicate, member)
                    )
            train_jobs.append(TrainJob("fitted_physics", primary_world, budget, replicate, 0))
        for replicate in prediction_replicates:
            for method in ADAPTED_METHODS:
                prediction_jobs.append(
                    PredictionJob(
                        job_id=content_id(
                            "prediction_job",
                            {
                                "schema": 1,
                                "method": method,
                                "world_id": primary_world,
                                "budget": budget,
                                "replicate": replicate,
                                "root_seed": contract.numerics.root_seed,
                            },
                        ),
                        method=method,
                        world_id=primary_world,
                        budget=budget,
                        replicate=replicate,
                    )
                )

    dataset_jobs = tuple(
        (primary_world, replicate) for replicate in sorted(needed_replicates)
    )
    prediction_set_jobs = dataset_jobs

    # Control rows.
    control_jobs: list[ControlJob] = []
    primary_panel = f"primary_{len(primary_replicates)}x{len(all_scenarios)}"
    curve_panel = f"budget_curve_{len(curve_pipelines)}x{len(curve_scenarios)}"
    for replicate in primary_replicates:
        for scenario_id in all_scenarios:
            in_curve = replicate in curve_pipelines and scenario_id in curve_scenarios
            panels = (
                (primary_panel, curve_panel) if in_curve else (primary_panel,)
            )
            for method in evaluation.all_reference_methods:
                control_jobs.append(
                    _control_job(
                        contract, method, primary_world, primary_budget, replicate,
                        scenario_id, panels,
                    )
                )
    for budget in evaluation.required_control_budgets:
        if budget == primary_budget:
            continue
        for replicate in curve_pipelines:
            for scenario_id in curve_scenarios:
                for method in ADAPTED_METHODS:
                    control_jobs.append(
                        _control_job(
                            contract, method, primary_world, budget, replicate,
                            scenario_id, (curve_panel,),
                        )
                    )

    # Seeded block-randomized execution order with method rotation.
    blocks: dict[tuple[str, int, int, str], list[ControlJob]] = {}
    for job in control_jobs:
        blocks.setdefault(
            (job.world_id, job.budget, job.replicate, job.scenario_id), []
        ).append(job)
    block_keys = sorted(blocks)
    rng = numpy_generator(contract.numerics.root_seed, "evaluation_order")
    permutation = rng.permutation(len(block_keys))
    ordered: list[ControlJob] = []
    for position, block_index in enumerate(permutation):
        block = blocks[block_keys[int(block_index)]]
        rotation = position % len(block)
        ordered.extend(block[rotation:] + block[:rotation])

    return ExecutionManifest(
        dataset_jobs=dataset_jobs,
        prediction_set_jobs=prediction_set_jobs,
        train_jobs=tuple(train_jobs),
        prediction_jobs=tuple(prediction_jobs),
        control_jobs=tuple(ordered),
    )
