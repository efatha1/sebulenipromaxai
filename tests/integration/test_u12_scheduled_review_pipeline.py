"""Integration test for the U12 scheduled review pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from api.dependencies import ActiveModelManifest, load_active_model_manifest, save_active_model_manifest
from training.config_loader import validate_config
from training.evaluate import CandidateModelMetadata, EvaluationSummary, FoldEvaluationSummary
from training.retraining import load_retraining_requests, request_retraining
from training.review import approve_candidate
from training.scheduler import run_scheduled_review


def _build_config(output_dir: Path):
    return validate_config(
        {
            "instrument": {"instrument_id": "TEST_INSTRUMENT"},
            "data_source": {"ohlc_path": "data/test.csv"},
            "time": {
                "source_timezone": "UTC",
                "runtime_timezone": "UTC",
                "daily_close": {"time": "17:00", "timezone": "America/New_York"},
            },
            "sessions": {
                "calendar_name": "default",
                "weekend_policy": "include",
                "holiday_policy": "include",
                "definitions": [{"name": "primary", "start": "00:00", "end": "23:59"}],
            },
            "resampling": {"base_timeframe": "1m", "target_timeframes": ["5m", "15m", "1h", "4h", "1d"]},
            "features": {"enabled_features": ["returns"], "deterministic_derived_features": ["returns"]},
            "labeling": {"thresholds": [10.0], "horizon_bars": [3, 5]},
            "walk_forward": {"train_bars": 4, "validation_bars": 2, "test_bars": 2, "step_bars": 2},
            "training": {
                "random_seed": 7,
                "batch_size": 2,
                "learning_rate": 0.001,
                "max_epochs": 1,
                "device_preference": "cpu",
                "allow_nondeterministic": False,
            },
            "retrieval": {"top_k_analogs": 2},
            "api": {"host": "127.0.0.1", "port": 8000},
            "reporting": {"output_dir": str(output_dir)},
            "schedules": {"reporting_cron": "5 9 * * *", "retraining_review_cron": "0 9 * * *"},
        }
    )


def _build_summary(tmp_path: Path) -> EvaluationSummary:
    candidate_path = tmp_path / "candidate.pt"
    candidate_path.write_text("candidate", encoding="utf-8")
    fold_summary = FoldEvaluationSummary(
        fold_id=0,
        model_id="candidate-model",
        config_hash="cfg123",
        checkpoint_path=candidate_path,
        artifact_dir=tmp_path,
        metrics_log_path=tmp_path / "metrics.jsonl",
        train_metrics={"total_loss": 1.0},
        validation_metrics={"total_loss": 0.4},
        test_metrics={"total_loss": 0.45},
        train_range=(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc)),
        validation_range=(datetime(2026, 1, 3, tzinfo=timezone.utc), datetime(2026, 1, 4, tzinfo=timezone.utc)),
        test_range=(datetime(2026, 1, 5, tzinfo=timezone.utc), datetime(2026, 1, 6, tzinfo=timezone.utc)),
    )
    candidate = CandidateModelMetadata(
        model_id="candidate-model",
        instrument_id="TEST_INSTRUMENT",
        fold_id=0,
        config_hash="cfg123",
        artifact_path=candidate_path,
        metrics={"total_loss": 0.4},
        training_range=(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc)),
    )
    return EvaluationSummary(
        fold_summaries=(fold_summary,),
        aggregate_metrics={"validation_total_loss": 0.4},
        candidate_model=candidate,
    )


def test_u12_scheduled_review_pipeline_requires_explicit_approval(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("current", encoding="utf-8")
    manifest_path = tmp_path / "active_model_manifest.json"
    save_active_model_manifest(
        manifest_path,
        ActiveModelManifest(
            checkpoint_path=str(current_checkpoint),
            lookbacks_by_timeframe={"1m": 1, "5m": 1, "15m": 1, "1h": 1, "4h": 1, "1d": 1},
            retrieval_memory=tuple(),
        ),
    )
    manual_request = request_retraining(
        output_dir=config.reporting.output_dir,
        instrument_id=config.instrument.instrument_id,
        reason="manual review requested",
        candidate_model_id="candidate-model",
        source="manual",
        requested_at=datetime(2026, 1, 1, 8, 30, tzinfo=timezone.utc),
    )
    summary = _build_summary(tmp_path)

    review_result = run_scheduled_review(
        config=config,
        current_time=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
        prediction_responses=tuple(),
        candidate_summary=summary,
    )

    manifest_after_review = load_active_model_manifest(manifest_path)
    reviewed_requests = load_retraining_requests(config.reporting.output_dir)
    assert review_result.retraining_review_due is True
    assert review_result.review_artifact_path is not None
    assert review_result.review_artifact_path.exists()
    assert manual_request.request_id in review_result.processed_request_ids
    assert manifest_after_review.checkpoint_path == str(current_checkpoint)
    assert reviewed_requests[0].status == "reviewed"

    approval = approve_candidate(
        candidate_summary=summary,
        active_model_manifest_path=manifest_path,
        lookbacks_by_timeframe={"1m": 1, "5m": 1, "15m": 1, "1h": 1, "4h": 1, "1d": 1},
        retrieval_memory=tuple(),
        output_dir=config.reporting.output_dir,
        approved=True,
        review_reason="explicit approval after scheduled review",
        reviewer_id="reviewer-1",
        reviewed_at=datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc),
    )

    manifest_after_approval = load_active_model_manifest(manifest_path)
    assert approval.status == "approved"
    assert manifest_after_approval.checkpoint_path == str(summary.candidate_model.artifact_path)
