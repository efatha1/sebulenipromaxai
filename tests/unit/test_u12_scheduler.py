"""Unit tests for U12 schedule dispatch."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from training.config_loader import validate_config
from training.contracts import AnalogRecordContract, PredictionRecordContract, PredictionResponseContract
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
            "schedules": {"reporting_cron": "0 9 * * *", "retraining_review_cron": "0 9 * * *"},
        }
    )


def test_u12_scheduler_runs_due_reporting_job(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    response = PredictionResponseContract(
        prediction=PredictionRecordContract(
            request_id="pred-1",
            reference_ts=datetime(2026, 1, 1, 8, 59, tzinfo=timezone.utc),
            horizon=3,
            event_probability=0.75,
            confidence=0.6,
            low_price=98.0,
            high_price=102.0,
            start_estimate=1,
            maturity_estimate=2,
            duration_estimate=2,
            low_confidence_advisory=False,
        ),
        top_k_analogs=(
            AnalogRecordContract(
                analog_id="a1",
                reference_ts=datetime(2026, 1, 1, 8, 55, tzinfo=timezone.utc),
                distance=0.1,
                outcome_summary="event_observed=1",
            ),
        ),
        summary_statistics={"analog_count": 1.0, "mean_distance": 0.1},
        grounded_natural_language_explanation="The top 1 training analog grounds this prediction.",
    )

    result = run_scheduled_review(
        config=config,
        current_time=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
        prediction_responses=(response,),
        candidate_summary=None,
    )

    assert result.reporting_due is True
    assert result.report_artifact_path is not None
    assert result.report_artifact_path.exists()
