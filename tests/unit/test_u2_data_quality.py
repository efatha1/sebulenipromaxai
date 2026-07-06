"""Unit tests for U2 data quality validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from training.config_loader import validate_config
from training.data_quality import DataQualityError, validate_bar_sequence


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
                "weekend_policy": "exclude",
                "holiday_policy": "include",
                "definitions": [{"name": "primary", "start": "00:00", "end": "23:59"}],
            },
            "resampling": {"base_timeframe": "1m", "target_timeframes": ["5m", "15m", "1h", "4h", "1d"]},
            "features": {
                "enabled_features": ["returns"],
                "deterministic_derived_features": ["returns"],
            },
            "labeling": {"thresholds": [10.0], "horizon_bars": [15]},
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


def build_frame(timestamps: list[datetime]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "end_ts": timestamps,
            "open": [1.0] * len(timestamps),
            "high": [2.0] * len(timestamps),
            "low": [0.5] * len(timestamps),
            "close": [1.5] * len(timestamps),
        }
    )
    frame.set_index("end_ts", inplace=True, drop=False)
    return frame


def test_duplicate_timestamp_fails() -> None:
    config = build_config()
    ts = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    frame = build_frame([ts, ts])

    with pytest.raises(DataQualityError, match="Duplicate timestamp"):
        validate_bar_sequence(frame, config)


def test_missing_bar_fails() -> None:
    config = build_config()
    frame = build_frame(
        [
            datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 0, 3, tzinfo=timezone.utc),
        ]
    )

    with pytest.raises(DataQualityError, match="Missing 1m bars"):
        validate_bar_sequence(frame, config)

