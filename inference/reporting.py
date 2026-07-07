"""U11 batch reporting and summary serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.contracts import PredictionResponseContract
from training.evaluate import CandidateModelMetadata, EvaluationSummary, FoldEvaluationSummary


class ReportingError(ValueError):
    """Raised when report generation or serialization fails."""


@dataclass(frozen=True)
class ReportArtifact:
    """Metadata for a generated batch report."""

    report_id: str
    report_type: str
    output_path: Path
    generated_at: datetime
    summary: dict[str, float]


def write_prediction_batch_report(
    *,
    predictions: tuple[PredictionResponseContract, ...] | list[PredictionResponseContract],
    output_dir: str | Path,
    report_name: str | None = None,
) -> ReportArtifact:
    """Write a batch prediction report from approved prediction outputs only."""
    responses = tuple(predictions)
    if not responses:
        raise ReportingError("predictions must not be empty.")

    generated_at = datetime.now(timezone.utc)
    report_id = report_name or f"prediction-report-{generated_at.strftime('%Y%m%d%H%M%S')}"
    output_path = Path(output_dir) / "batch_reports" / f"{report_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    advisory_count = sum(1 for response in responses if response.prediction.low_confidence_advisory)
    summary = {
        "prediction_count": float(len(responses)),
        "advisory_count": float(advisory_count),
        "mean_event_probability": float(
            sum(response.prediction.event_probability for response in responses) / len(responses)
        ),
    }
    payload = {
        "report_id": report_id,
        "report_type": "prediction_batch",
        "generated_at": generated_at.isoformat(),
        "summary": summary,
        "predictions": [serialize_prediction_response(response) for response in responses],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return ReportArtifact(
        report_id=report_id,
        report_type="prediction_batch",
        output_path=output_path,
        generated_at=generated_at,
        summary=summary,
    )


def write_evaluation_summary(
    *,
    summary: EvaluationSummary,
    output_path: str | Path,
) -> Path:
    """Serialize an evaluation summary to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_evaluation_summary(summary)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def serialize_prediction_response(response: PredictionResponseContract) -> dict[str, Any]:
    """Serialize a typed prediction response into JSON-friendly data."""
    return response.model_dump(mode="json")


def serialize_evaluation_summary(summary: EvaluationSummary) -> dict[str, Any]:
    """Serialize an evaluation summary into JSON-friendly data."""
    return {
        "aggregate_metrics": summary.aggregate_metrics,
        "candidate_model": _serialize_candidate(summary.candidate_model),
        "fold_summaries": [_serialize_fold_summary(item) for item in summary.fold_summaries],
    }


def _serialize_candidate(candidate: CandidateModelMetadata) -> dict[str, Any]:
    return {
        "model_id": candidate.model_id,
        "instrument_id": candidate.instrument_id,
        "fold_id": int(candidate.fold_id),
        "config_hash": candidate.config_hash,
        "artifact_path": str(candidate.artifact_path),
        "metrics": candidate.metrics,
        "training_range": [candidate.training_range[0].isoformat(), candidate.training_range[1].isoformat()],
    }


def _serialize_fold_summary(summary: FoldEvaluationSummary) -> dict[str, Any]:
    return {
        "fold_id": int(summary.fold_id),
        "model_id": summary.model_id,
        "config_hash": summary.config_hash,
        "checkpoint_path": str(summary.checkpoint_path),
        "artifact_dir": str(summary.artifact_dir),
        "metrics_log_path": str(summary.metrics_log_path),
        "train_metrics": summary.train_metrics,
        "validation_metrics": summary.validation_metrics,
        "test_metrics": summary.test_metrics,
        "train_range": [summary.train_range[0].isoformat(), summary.train_range[1].isoformat()],
        "validation_range": [summary.validation_range[0].isoformat(), summary.validation_range[1].isoformat()],
        "test_range": [summary.test_range[0].isoformat(), summary.test_range[1].isoformat()],
    }
