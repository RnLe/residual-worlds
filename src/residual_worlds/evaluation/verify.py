"""Completeness verification for an expanded execution manifest."""

from __future__ import annotations

from dataclasses import dataclass

from residual_worlds.evaluation.manifest import ExecutionManifest
from residual_worlds.paths import evaluation_dir, prediction_job_dir
from residual_worlds.provenance import ArtifactError, is_complete, verify_artifact


@dataclass(frozen=True)
class CompletenessReport:
    control_expected: int
    control_complete: int
    prediction_expected: int
    prediction_complete: int
    missing: tuple[str, ...]
    corrupt: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing and not self.corrupt


def verify_complete(manifest: ExecutionManifest) -> CompletenessReport:
    """Every scheduled row must have a checksum-valid complete artifact."""
    missing: list[str] = []
    corrupt: list[str] = []
    control_complete = 0
    for job in manifest.control_jobs:
        directory = evaluation_dir(job.job_id)
        if not is_complete(directory):
            missing.append(f"control:{job.job_id}")
            continue
        try:
            verify_artifact(directory)
            control_complete += 1
        except ArtifactError:
            corrupt.append(f"control:{job.job_id}")
    prediction_complete = 0
    for prediction in manifest.prediction_jobs:
        directory = prediction_job_dir(prediction.job_id)
        if not is_complete(directory):
            missing.append(f"prediction:{prediction.job_id}")
            continue
        try:
            verify_artifact(directory)
            prediction_complete += 1
        except ArtifactError:
            corrupt.append(f"prediction:{prediction.job_id}")
    return CompletenessReport(
        control_expected=len(manifest.control_jobs),
        control_complete=control_complete,
        prediction_expected=len(manifest.prediction_jobs),
        prediction_complete=prediction_complete,
        missing=tuple(missing),
        corrupt=tuple(corrupt),
    )
