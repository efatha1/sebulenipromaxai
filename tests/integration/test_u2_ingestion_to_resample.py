"""Integration tests for U2 ingestion -> calendar -> quality -> resample."""

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
from training.resample import resample_timeframes


def test_u2_end_to_end_csv_to_multitimeframe_output(tmp_path: Path) -> None:
    ohlc_path = tmp_path / "ohlc.csv"
    config_path = tmp_path / "config.yaml"

    ny_tz = ZoneInfo("America/New_York")

    start_ny = datetime(2026, 1, 2, 16, 55, tzinfo=ny_tz)
    timestamps = [(start_ny + pd.Timedelta(minutes=i)).astimezone(timezone.utc) for i in range(0, 11)]

    df = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in timestamps],
            "open": [1.0] * len(timestamps),
            "high": [2.0] * len(timestamps),
            "low": [0.5] * len(timestamps),
            "close": [1.5] * len(timestamps),
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
        "features": {"enabled_features": ["returns"], "deterministic_derived_features": ["returns"]},
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
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    config = load_config(config_path)
    frame_1m = load_ohlc_frame(config)
    normalize_calendar(frame_1m, config)
    validate_bar_sequence(frame_1m, config)
    out = resample_timeframes(frame_1m, config)

    assert set(out.keys()) == {"1m", "5m", "15m", "1h", "4h", "1d"}
    assert len(out["1m"]) == len(timestamps)
    assert out["1m"]["end_ts"].dt.tz is not None
    assert out["5m"]["end_ts"].dt.tz is not None
    assert out["1d"]["end_ts"].dt.tz is not None
    assert len(out["1d"]) == 2

