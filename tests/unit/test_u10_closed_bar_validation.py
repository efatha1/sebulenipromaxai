"""Unit tests for U10 closed-bar validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from inference.runtime_features import RuntimeFeatureError, build_runtime_window
from training.config_loader import validate_config
from training.contracts import PredictionRequestContract


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
            "features": {
                "enabled_features": ["returns", "calendar_time"],
                "deterministic_derived_features": ["returns", "calendar_time"],
            },
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
            "schedules": {"reporting_cron": "0 9 * * *", "retraining_review_cron": "0 0 * * 1"},
        }
    )


def _build_request() -> PredictionRequestContract:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    bars = [
        {
            "timestamp": (start + timedelta(minutes=index)).isoformat(),
            "open": 100.0 + index,
            "high": 100.5 + index,
            "low": 99.5 + index,
            "close": 100.25 + index,
        }
        for index in range(6)
    ]
    return PredictionRequestContract.model_validate(
        {
            "instrument_id": "TEST_INSTRUMENT",
            "bars_1m": bars,
            "horizon_mode": "single",
            "horizon_bars": 3,
            "threshold": 10.0,
            "top_k_analogs": 1,
        }
    )


def test_u10_rejects_incomplete_recent_bar(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    request = _build_request()
    latest_ts = request.bars_1m[-1].timestamp

    with pytest.raises(RuntimeFeatureError, match="inference must reject incomplete recent bars"):
        build_runtime_window(
            request,
            config,
            {timeframe: 1 for timeframe in ("1m", "5m", "15m", "1h", "4h", "1d")},
            current_time=latest_ts - timedelta(seconds=1),
        )
