"""Integration tests for U1 config loading into entry points."""

from __future__ import annotations

from pathlib import Path

import yaml

from training.entrypoints import load_inference_config, load_training_config


def build_valid_config() -> dict[str, object]:
    """Create a valid config payload for tests."""
    return {
        "instrument": {"instrument_id": "TEST_INSTRUMENT"},
        "data_source": {"ohlc_path": "data/test.parquet"},
        "time": {
            "source_timezone": "UTC",
            "runtime_timezone": "America/New_York",
            "daily_close": {"time": "17:00", "timezone": "America/New_York"},
        },
        "sessions": {
            "calendar_name": "default",
            "weekend_policy": "exclude",
            "holiday_policy": "exclude",
            "definitions": [{"name": "primary", "start": "00:00", "end": "23:59"}],
        },
        "resampling": {
            "base_timeframe": "1m",
            "target_timeframes": ["5m", "15m", "1h", "4h", "1d"],
        },
        "features": {
            "enabled_features": ["returns", "ranges", "wick_body_ratios"],
            "deterministic_derived_features": ["returns", "ranges", "wick_body_ratios"],
        },
        "labeling": {"thresholds": [10.0], "horizon_bars": [15, 60, 240]},
        "walk_forward": {
            "train_bars": 1000,
            "validation_bars": 200,
            "test_bars": 200,
            "step_bars": 200,
        },
        "training": {
            "random_seed": 42,
            "batch_size": 32,
            "learning_rate": 0.001,
            "max_epochs": 5,
            "device_preference": "cpu",
            "allow_nondeterministic": False,
        },
        "retrieval": {"top_k_analogs": 5},
        "api": {"host": "127.0.0.1", "port": 8000},
        "reporting": {"output_dir": "artifacts/reports"},
        "schedules": {
            "reporting_cron": "0 9 * * *",
            "retraining_review_cron": "0 0 * * 1",
        },
    }


def test_training_and_inference_entrypoints_share_the_same_config_contract(tmp_path: Path) -> None:
    """Training and inference entry points should load the same validated config."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(build_valid_config(), sort_keys=False), encoding="utf-8")

    env = {"SEBULENI__INSTRUMENT__INSTRUMENT_ID": "OVERRIDDEN_INTEGRATION"}

    training_config = load_training_config(config_path, env=env)
    inference_config = load_inference_config(config_path, env=env)

    assert training_config == inference_config
    assert training_config.instrument.instrument_id == "OVERRIDDEN_INTEGRATION"
    assert inference_config.resampling.target_timeframes == ("5m", "15m", "1h", "4h", "1d")

