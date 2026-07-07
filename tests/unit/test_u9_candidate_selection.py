"""Unit tests for U9 candidate selection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from training.checkpointing import stable_config_hash
from training.config_loader import validate_config
from training.evaluate import FoldEvaluationSummary, select_candidate_model


def _build_config(output_dir: Path) -> object:
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
            "labeling": {"thresholds": [10.0], "horizon_bars": [5]},
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
            "schedules": {"reporting_cron": "0 9 * * *", "retraining_review_cron": "0 0 * * 1"},
        }
    )


def test_u9_candidate_selection_uses_validation_then_test_then_fold_id(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    config_hash = stable_config_hash(config)
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)

    fold_a = FoldEvaluationSummary(
        fold_id=1,
        model_id="model-a",
        config_hash=config_hash,
        checkpoint_path=tmp_path / "a.pt",
        artifact_dir=tmp_path,
        metrics_log_path=tmp_path / "a.jsonl",
        train_metrics={"total_loss": 1.0},
        validation_metrics={"total_loss": 0.50},
        test_metrics={"total_loss": 0.45},
        train_range=(start, end),
        validation_range=(start, end),
        test_range=(start, end),
    )
    fold_b = FoldEvaluationSummary(
        fold_id=0,
        model_id="model-b",
        config_hash=config_hash,
        checkpoint_path=tmp_path / "b.pt",
        artifact_dir=tmp_path,
        metrics_log_path=tmp_path / "b.jsonl",
        train_metrics={"total_loss": 1.0},
        validation_metrics={"total_loss": 0.50},
        test_metrics={"total_loss": 0.40},
        train_range=(start, end),
        validation_range=(start, end),
        test_range=(start, end),
    )

    candidate = select_candidate_model(config=config, fold_summaries=(fold_a, fold_b))

    assert candidate.model_id == "model-b"
    assert candidate.fold_id == 0
    assert candidate.artifact_path == tmp_path / "b.pt"
