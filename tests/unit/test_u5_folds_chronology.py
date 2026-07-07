"""Unit tests for U5 fold chronology."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from training.config_loader import validate_config
from training.folds import build_walk_forward_folds, validate_temporal_isolation


def build_config() -> object:
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
            "walk_forward": {"train_bars": 10, "validation_bars": 5, "test_bars": 5, "step_bars": 5},
            "training": {
                "random_seed": 1,
                "batch_size": 1,
                "learning_rate": 0.001,
                "max_epochs": 1,
                "device_preference": "cpu",
                "allow_nondeterministic": False,
            },
            "retrieval": {"top_k_analogs": 1},
            "api": {"host": "127.0.0.1", "port": 8000},
            "reporting": {"output_dir": "artifacts"},
            "schedules": {"reporting_cron": "0 9 * * *", "retraining_review_cron": "0 0 * * 1"},
        }
    )


def test_fold_chronology_ordering() -> None:
    config = build_config()
    reference_ts = [datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc) for i in range(40)]
    labeled = pd.DataFrame({"reference_ts": reference_ts, "ambiguous": [False] * len(reference_ts)})

    folds = build_walk_forward_folds(labeled, config)
    assert len(folds) > 0

    validate_temporal_isolation(folds, pd.DatetimeIndex(reference_ts))

    first = folds[0]
    assert first.train.start_index == 0
    assert first.train.end_index == 10
    assert first.metadata["purge_gap_bars"] == 5
    assert first.validation.start_index == 15
    assert first.test.start_index == 25
