"""U9 evaluation summaries and candidate selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from training.checkpointing import stable_config_hash
from training.config_schema import RuntimeConfig
from training.metrics import aggregate_metric_dicts

LOGGER = logging.getLogger(__name__)


class EvaluationError(ValueError):
    """Raised when evaluation summary generation fails."""


@dataclass(frozen=True)
class FoldEvaluationSummary:
    """Stable per-fold evaluation summary."""

    fold_id: int
    model_id: str
    config_hash: str
    checkpoint_path: Path
    artifact_dir: Path
    metrics_log_path: Path
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    train_range: tuple[datetime, datetime]
    validation_range: tuple[datetime, datetime]
    test_range: tuple[datetime, datetime]


@dataclass(frozen=True)
class CandidateModelMetadata:
    """Selected candidate model metadata."""

    model_id: str
    instrument_id: str
    fold_id: int
    config_hash: str
    artifact_path: Path
    metrics: dict[str, float]
    training_range: tuple[datetime, datetime]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation result for a full walk-forward run."""

    fold_summaries: tuple[FoldEvaluationSummary, ...]
    aggregate_metrics: dict[str, float]
    candidate_model: CandidateModelMetadata


def select_candidate_model(
    *,
    config: RuntimeConfig,
    fold_summaries: tuple[FoldEvaluationSummary, ...] | list[FoldEvaluationSummary],
) -> CandidateModelMetadata:
    """Select the candidate model deterministically from evaluated folds."""
    summaries = tuple(fold_summaries)
    if not summaries:
        raise EvaluationError("fold_summaries must not be empty.")

    expected_hash = stable_config_hash(config)
    for summary in summaries:
        if summary.config_hash != expected_hash:
            raise EvaluationError("fold summary config_hash mismatch.")

    best = min(
        summaries,
        key=lambda summary: (
            summary.validation_metrics["total_loss"],
            summary.test_metrics["total_loss"],
            summary.fold_id,
        ),
    )

    LOGGER.info(
        "selected_candidate_model",
        extra={
            "event": "selected_candidate_model",
            "model_id": best.model_id,
            "fold_id": int(best.fold_id),
            "validation_total_loss": float(best.validation_metrics["total_loss"]),
            "test_total_loss": float(best.test_metrics["total_loss"]),
        },
    )
    return CandidateModelMetadata(
        model_id=best.model_id,
        instrument_id=config.instrument.instrument_id,
        fold_id=best.fold_id,
        config_hash=best.config_hash,
        artifact_path=best.checkpoint_path,
        metrics=dict(best.validation_metrics),
        training_range=best.train_range,
    )


def run_evaluation(
    *,
    config: RuntimeConfig,
    fold_summaries: tuple[FoldEvaluationSummary, ...] | list[FoldEvaluationSummary],
) -> EvaluationSummary:
    """Aggregate fold summaries into an evaluation result."""
    summaries = tuple(fold_summaries)
    if not summaries:
        raise EvaluationError("fold_summaries must not be empty.")

    aggregate_metrics: dict[str, float] = {}
    for split_name, attribute in (
        ("train", "train_metrics"),
        ("validation", "validation_metrics"),
        ("test", "test_metrics"),
    ):
        split_metrics = [getattr(summary, attribute) for summary in summaries]
        for metric_name, value in aggregate_metric_dicts(split_metrics).items():
            aggregate_metrics[f"{split_name}_{metric_name}"] = float(value)
    aggregate_metrics["fold_count"] = float(len(summaries))

    candidate = select_candidate_model(config=config, fold_summaries=summaries)
    LOGGER.info(
        "completed_run_evaluation",
        extra={
            "event": "completed_run_evaluation",
            "fold_count": len(summaries),
            "candidate_model_id": candidate.model_id,
            "config_hash": stable_config_hash(config),
        },
    )
    return EvaluationSummary(
        fold_summaries=summaries,
        aggregate_metrics=aggregate_metrics,
        candidate_model=candidate,
    )
