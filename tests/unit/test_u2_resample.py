"""Unit tests for U2 deterministic resampling."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from training.config_loader import validate_config
from training.resample import resample_timeframes


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


def test_resample_5m_ohlc_aggregation() -> None:
    config = build_config()
    timestamps = [datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc) for i in range(1, 11)]
    frame = pd.DataFrame(
        {
            "end_ts": timestamps,
            "open": list(range(10)),
            "high": [10.0 + i for i in range(10)],
            "low": [-1.0 - i for i in range(10)],
            "close": list(range(100, 110)),
        }
    )
    frame.set_index("end_ts", inplace=True, drop=False)

    out = resample_timeframes(frame, config)
    bars_5m = out["5m"]

    first = bars_5m.iloc[0].to_dict()
    assert first["timeframe"] == "5m"
    assert first["open"] == 0
    assert first["close"] == 104
    assert first["high"] == 14.0
    assert first["low"] == -5.0

