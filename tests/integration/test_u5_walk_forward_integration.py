"""Integration tests for U5 walk-forward fold generation."""

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
from training.folds import build_walk_forward_folds, validate_temporal_isolation
from training.labeling import generate_labels
from training.resample import resample_timeframes


def test_u5_generates_reproducible_folds(tmp_path: Path) -> None:
    ohlc_path = tmp_path / "ohlc.csv"
    config_path = tmp_path / "config.yaml"

    ny_tz = ZoneInfo("America/New_York")
    start_ny = datetime(2026, 1, 2, 16, 0, tzinfo=ny_tz)
    timestamps = [(start_ny + pd.Timedelta(minutes=i)).astimezone(timezone.utc) for i in range(0, 120)]

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
        "labeling": {"thresholds": [10.0], "horizon_bars": [15]},
        "walk_forward": {"train_bars": 40, "validation_bars": 20, "test_bars": 20, "step_bars": 20},
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
    bars_1m = load_ohlc_frame(config)
    normalize_calendar(bars_1m, config)
    validate_bar_sequence(bars_1m, config)
    bars_by_tf = resample_timeframes(bars_1m, config)
    build_features(bars_by_tf, config)

    labels = generate_labels(bars_1m, config, horizon_mode="single", horizon_bars=15)
    label_df = labels[(15, 10.0)]
    label_df = label_df[~label_df["ambiguous"]].copy()

    folds1 = build_walk_forward_folds(label_df, config)
    folds2 = build_walk_forward_folds(label_df, config)
    assert folds1 == folds2

    validate_temporal_isolation(folds1, pd.DatetimeIndex(label_df["reference_ts"]))
