"""Unit tests for U2 timezone normalization and daily close bucketing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yaml

from training.config_loader import load_config
from training.data_loader import OhlcLoadError, load_ohlc_frame
from training.resample import resample_timeframes


def test_daily_close_bucketing_creates_two_days(tmp_path: Path) -> None:
    ohlc_path = tmp_path / "ohlc.csv"
    config_path = tmp_path / "config.yaml"

    runtime_tz = "America/New_York"
    ny_tz = ZoneInfo(runtime_tz)

    ny_times = [
        datetime(2026, 1, 2, 16, 58, tzinfo=ny_tz).astimezone(timezone.utc),
        datetime(2026, 1, 2, 17, 0, tzinfo=ny_tz).astimezone(timezone.utc),
        datetime(2026, 1, 2, 17, 2, tzinfo=ny_tz).astimezone(timezone.utc),
    ]

    df = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in ny_times],
            "open": [1.0, 1.0, 1.0],
            "high": [2.0, 2.0, 2.0],
            "low": [0.5, 0.5, 0.5],
            "close": [1.5, 1.5, 1.5],
        }
    )
    df.to_csv(ohlc_path, index=False)

    config_path.write_text(
        yaml.safe_dump(
            {
                "instrument": {"instrument_id": "TEST_INSTRUMENT"},
                "data_source": {"ohlc_path": str(ohlc_path)},
                "time": {
                    "source_timezone": "UTC",
                    "runtime_timezone": runtime_tz,
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    frame = load_ohlc_frame(config)
    out = resample_timeframes(frame, config)

    daily = out["1d"]
    assert len(daily) == 2


def test_load_ohlc_frame_rejects_extra_columns(tmp_path: Path) -> None:
    ohlc_path = tmp_path / "ohlc.csv"
    df = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat()],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "extra": [123],
        }
    )
    df.to_csv(ohlc_path, index=False)

    config = load_config(Path("config") / "base.yaml")
    config = config.model_copy(update={"data_source": config.data_source.model_copy(update={"ohlc_path": ohlc_path})})

    with pytest.raises(OhlcLoadError, match="Extra columns"):
        load_ohlc_frame(config)
