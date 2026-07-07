"""Unit tests for U4 event start and maturity semantics."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from training.config_loader import validate_config
from training.labeling import generate_labels


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
            "walk_forward": {"train_bars": 1000, "validation_bars": 200, "test_bars": 200, "step_bars": 200},
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


def test_event_start_and_maturity_offsets() -> None:
    config = build_config()
    end_ts = [datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc) for i in range(8)]
    close = [100.0] * 8
    high = [100.0, 105.0, 111.0, 112.0, 104.0, 103.0, 100.0, 100.0]
    low = [100.0, 99.0, 98.0, 97.0, 96.0, 100.0, 100.0, 100.0]
    bars = pd.DataFrame({"end_ts": end_ts, "open": close, "high": high, "low": low, "close": close})

    labels = generate_labels(bars, config, horizon_mode="single", horizon_bars=5)
    df = labels[(5, 10.0)]

    row0 = df.iloc[0]
    assert row0["event_start_offset"] == 2
    assert row0["maturity_offset"] == 3
