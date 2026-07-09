"""U12 schedule runner for reporting and retraining review."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from inference.reporting import ReportArtifact, write_prediction_batch_report
from training.config_schema import RuntimeConfig
from training.contracts import PredictionResponseContract
from training.evaluate import EvaluationSummary
from training.retraining import (
    load_retraining_requests,
    request_retraining,
    update_retraining_request_status,
)
from training.review import write_review_recommendation

LOGGER = logging.getLogger(__name__)


class SchedulerError(ValueError):
    """Raised when scheduled job execution fails."""


@dataclass(frozen=True)
class ScheduledReviewResult:
    """Typed result of a scheduled review run."""

    reporting_due: bool
    retraining_review_due: bool
    report_artifact_path: Path | None
    review_artifact_path: Path | None
    processed_request_ids: tuple[str, ...]


def run_scheduled_review(
    *,
    config: RuntimeConfig,
    current_time: datetime,
    prediction_responses: tuple[PredictionResponseContract, ...] | list[PredictionResponseContract] | None = None,
    candidate_summary: EvaluationSummary | None = None,
) -> ScheduledReviewResult:
    """Run scheduled reporting and retraining review without auto-promotion."""
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise SchedulerError("current_time must be timezone-aware.")

    reporting_due = _matches_cron(config.schedules.reporting_cron, current_time)
    retraining_due = _matches_cron(config.schedules.retraining_review_cron, current_time)

    report_artifact: ReportArtifact | None = None
    if reporting_due:
        responses = tuple(prediction_responses or tuple())
        if not responses:
            raise SchedulerError("prediction_responses are required when the reporting schedule is due.")
        report_artifact = write_prediction_batch_report(
            predictions=responses,
            output_dir=config.reporting.output_dir,
            report_name=f"scheduled-report-{current_time.strftime('%Y%m%d%H%M%S')}",
        )
        LOGGER.info(
            "scheduled_reporting_job_completed",
            extra={
                "event": "scheduled_reporting_job_completed",
                "report_artifact_path": str(report_artifact.output_path),
                "report_id": report_artifact.report_id,
            },
        )

    review_artifact_path: Path | None = None
    processed_request_ids: tuple[str, ...] = tuple()
    if retraining_due:
        pending_requests = load_retraining_requests(config.reporting.output_dir, statuses=("pending_review",))
        if not pending_requests and candidate_summary is not None:
            request_retraining(
                output_dir=config.reporting.output_dir,
                instrument_id=config.instrument.instrument_id,
                reason="scheduled_retraining_review",
                candidate_model_id=candidate_summary.candidate_model.model_id,
                source="scheduled",
                requested_at=current_time,
            )
            pending_requests = load_retraining_requests(config.reporting.output_dir, statuses=("pending_review",))

        request_ids = tuple(item.request_id for item in pending_requests)
        review_artifact_path = write_review_recommendation(
            output_dir=config.reporting.output_dir,
            candidate_summary=candidate_summary,
            request_ids=request_ids,
            recommended_action="manual_approval_required" if candidate_summary is not None else "review_inputs_missing",
            current_time=current_time,
            note="Scheduled retraining review completed without promotion.",
        )
        for request_record in pending_requests:
            update_retraining_request_status(
                request_record.output_path,
                status="reviewed",
                reviewed_at=current_time,
                review_note="Manual approval required; no auto-promotion performed.",
            )
        processed_request_ids = request_ids
        LOGGER.info(
            "scheduled_retraining_review_completed",
            extra={
                "event": "scheduled_retraining_review_completed",
                "review_artifact_path": str(review_artifact_path),
                "processed_request_ids": list(processed_request_ids),
                "candidate_model_id": (
                    candidate_summary.candidate_model.model_id if candidate_summary is not None else None
                ),
                "auto_promotion_performed": False,
            },
        )

    return ScheduledReviewResult(
        reporting_due=reporting_due,
        retraining_review_due=retraining_due,
        report_artifact_path=(report_artifact.output_path if report_artifact is not None else None),
        review_artifact_path=review_artifact_path,
        processed_request_ids=processed_request_ids,
    )


def _matches_cron(expression: str, current_time: datetime) -> bool:
    parts = expression.split()
    if len(parts) != 5:
        raise SchedulerError("cron expressions must contain exactly 5 fields.")
    minute, hour, day_of_month, month, day_of_week = parts
    values = (
        (minute, current_time.minute),
        (hour, current_time.hour),
        (day_of_month, current_time.day),
        (month, current_time.month),
        (day_of_week, (current_time.weekday() + 1) % 7),
    )
    return all(_match_cron_field(token, value) for token, value in values)


def _match_cron_field(token: str, value: int) -> bool:
    if token == "*":
        return True
    if token.startswith("*/"):
        step = int(token[2:])
        if step <= 0:
            raise SchedulerError("cron step values must be positive.")
        return value % step == 0
    return int(token) == value
