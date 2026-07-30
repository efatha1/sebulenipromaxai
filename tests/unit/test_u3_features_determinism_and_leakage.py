"""Unit tests for U3 determinism and leakage prevention."""

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
            "resampling": {"base_timeframe": "1m", "target_timeframes": ["5m", "15m", "1h", "4h"]},
            "features": {
                "enabled_features": ["returns", "ranges", "calendar_time", "session_primary", "roll_mean_return_3"],
                "deterministic_derived_features": ["returns", "ranges"],
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


def build_bars(n: int) -> pd.DataFrame:
    end_ts = [datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc) for i in range(n)]
    close = [100.0 + i for i in range(n)]
    frame = pd.DataFrame(
        {
            "end_ts": end_ts,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
        }
    )
    return frame


def test_determinism_identical_input_identical_output() -> None:
    config = build_config()
    bars = build_bars(10)
    bars_by_tf = {tf: bars.copy() for tf in ("1m", "5m", "15m", "1h", "4h")}

    out1 = build_features(bars_by_tf, config)
    out2 = build_features(bars_by_tf, config)

    for tf in out1:
        assert out1[tf].equals(out2[tf])


def test_no_future_leakage_from_future_bar_mutation() -> None:
    config = build_config()
    bars = build_bars(10)
    bars_by_tf = {tf: bars.copy() for tf in ("1m", "5m", "15m", "1h", "4h")}

    out1 = build_features(bars_by_tf, config)["1m"]

    mutated = bars.copy()
    mutated.loc[len(mutated) - 1, "close"] = mutated.loc[len(mutated) - 1, "close"] + 1000.0
    bars_by_tf_mut = {tf: mutated.copy() for tf in ("1m", "5m", "15m", "1h", "4h")}
    out2 = build_features(bars_by_tf_mut, config)["1m"]

    assert out1.iloc[:7].equals(out2.iloc[:7])

