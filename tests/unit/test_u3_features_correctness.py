"""Unit tests for U3 feature correctness."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from training.config_loader import validate_config
from training.features import build_features


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
                "enabled_features": ["returns", "ranges", "wick_body_ratios", "calendar_time", "session_primary"],
                "deterministic_derived_features": ["returns", "ranges", "wick_body_ratios"],
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


def build_bars() -> pd.DataFrame:
    end_ts = [
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
    ]
    frame = pd.DataFrame(
        {
            "end_ts": end_ts,
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
        }
    )
    return frame


def test_base_feature_values() -> None:
    config = build_config()
    bars = build_bars()
    bars_by_tf = {tf: bars.copy() for tf in ("1m", "5m", "15m", "1h", "4h", "1d")}

    out = build_features(bars_by_tf, config)
    features_1m = out["1m"]

    assert features_1m.iloc[0]["return"] == 0.0
    assert features_1m.iloc[1]["return"] == (102.0 / 101.0) - 1.0
    assert features_1m.iloc[0]["hl_range"] == 3.0
    assert features_1m.iloc[0]["body"] == 1.0
    assert features_1m.iloc[0]["upper_wick"] == 1.0
    assert features_1m.iloc[0]["lower_wick"] == 1.0
    assert features_1m.iloc[0]["session_primary"] == 1

