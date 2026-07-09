"""Integration tests for U4 label generation stability."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from training.calendar import normalize_calendar
from training.config_loader import load_config
from training.data_loader import load_ohlc_frame
from training.data_quality import validate_bar_sequence
from training.features import build_features
from training.labeling import generate_labels
from training.resample import resample_timeframes
from training.windowing import build_windows


def test_u4_labels_are_stable_across_runs(tmp_path: Path) -> None:
    ohlc_path = tmp_path / "ohlc.csv"
    config_path = tmp_path / "config.yaml"

    ny_tz = ZoneInfo("America/New_York")
    start_ny = datetime(2026, 1, 2, 16, 40, tzinfo=ny_tz)
    timestamps = [(start_ny + pd.Timedelta(minutes=i)).astimezone(timezone.utc) for i in range(0, 80)]

    close = [100.0] * len(timestamps)
    high = [100.0] * len(timestamps)
    low = [100.0] * len(timestamps)
    high[10] = 120.0
    low[30] = 80.0

    df = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in timestamps],
            "open": close,
            "high": high,
            "low": low,
            "close": close,
        }
    )
    df.to_csv(ohlc_path, index=False)

    config_payload = {
        "instrument": {"instrument_id": "TEST_INSTRUMENT"},
        "data_source": {"ohlc_path": str(ohlc_path)},
        "time": {
            "source_timezone": "UTC",
            "runtime_timezone": "America/New_York",
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
            "enabled_features": ["returns", "ranges", "wick_body_ratios", "calendar_time", "session_primary"],
            "deterministic_derived_features": ["returns", "ranges", "wick_body_ratios"],
        },
        "labeling": {"thresholds": [10.0], "horizon_bars": [15, 30]},
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
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    config = load_config(config_path)
    frame_1m = load_ohlc_frame(config)
    normalize_calendar(frame_1m, config)
    validate_bar_sequence(frame_1m, config)
    bars_by_tf = resample_timeframes(frame_1m, config)
    features_by_tf = build_features(bars_by_tf, config)
    build_windows(features_by_tf, lookbacks_by_timeframe={"1m": 5, "5m": 3, "15m": 2, "1h": 1, "4h": 1, "1d": 1})

    labels1 = generate_labels(frame_1m, config, horizon_mode="multi")
    labels2 = generate_labels(frame_1m, config, horizon_mode="multi")

    assert set(labels1.keys()) == set(labels2.keys())
    for key in labels1:
        assert labels1[key].equals(labels2[key])

