"""Repository-relative locations for generated artifacts.

All generated outputs live under one artifact root (``artifacts/`` by
default, or ``RW_ARTIFACT_ROOT`` if set). Directories are addressed by
content-derived identifiers, never by timestamps, so reruns either
reproduce an existing artifact or refuse to overwrite it.
"""

from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    """Locate the repository root by walking up from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repository root from " + str(here))


def artifact_root() -> Path:
    override = os.environ.get("RW_ARTIFACT_ROOT")
    root = Path(override) if override else repository_root() / "artifacts"
    return root


def verification_dir(verification_id: str) -> Path:
    return artifact_root() / "verification" / verification_id


def calibration_dir(name: str) -> Path:
    return artifact_root() / "calibration" / name


def dataset_dir(dataset_id: str) -> Path:
    return artifact_root() / "datasets" / dataset_id


def prediction_set_dir(prediction_set_id: str) -> Path:
    return artifact_root() / "prediction_sets" / prediction_set_id


def run_dir(run_id: str) -> Path:
    return artifact_root() / "runs" / run_id


def condition_dir(condition_id: str) -> Path:
    return artifact_root() / "model_conditions" / condition_id


def evaluation_dir(evaluation_job_id: str) -> Path:
    return artifact_root() / "evaluations" / evaluation_job_id


def prediction_job_dir(prediction_job_id: str) -> Path:
    return artifact_root() / "predictions" / prediction_job_id


def analysis_dir(analysis_id: str) -> Path:
    return artifact_root() / "analyses" / analysis_id


def figures_dir(analysis_id: str) -> Path:
    return artifact_root() / "figures" / analysis_id


def core_result_dir(core_id: str) -> Path:
    return artifact_root() / "core_results" / core_id


def media_dir(media_id: str) -> Path:
    return artifact_root() / "media" / media_id


def report_dir(report_id: str) -> Path:
    return artifact_root() / "reports" / report_id


def public_result_root() -> Path:
    return artifact_root() / "public_result"
