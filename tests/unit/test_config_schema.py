"""Unit tests for U1 config schema validation."""

from __future__ import annotations

from copy import deepcopy

import pytest

from training.config_loader import ConfigValidationError, validate_config
from training.config_schema import RuntimeConfig


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


def test_validate_config_returns_runtime_config() -> None:
    """A valid payload should yield an immutable runtime config."""
    config = validate_config(build_valid_config())

    assert isinstance(config, RuntimeConfig)
    assert config.instrument.instrument_id == "TEST_INSTRUMENT"
    assert config.resampling.target_timeframes == ("5m", "15m", "1h", "4h", "1d")


def test_validate_config_missing_required_field_fails() -> None:
    """Missing required config sections should fail fast."""
    invalid = build_valid_config()
    del invalid["api"]

    with pytest.raises(ConfigValidationError, match="api"):
        validate_config(invalid)


def test_validate_config_invalid_type_fails() -> None:
    """Invalid field types should fail fast with a useful path."""
    invalid = build_valid_config()
    invalid["training"] = deepcopy(invalid["training"])
    invalid["training"]["random_seed"] = "forty-two"

    with pytest.raises(ConfigValidationError, match="training.random_seed"):
        validate_config(invalid)


def test_runtime_config_is_immutable() -> None:
    """Validated runtime config objects should be immutable."""
    config = validate_config(build_valid_config())

    with pytest.raises(Exception):
        config.instrument.instrument_id = "MUTATED"


def test_validate_config_accepts_preprocessing_lookbacks() -> None:
    """Optional preprocessing lookbacks should validate when all timeframes are present."""
    payload = build_valid_config()
    payload["preprocessing"] = {
        "lookbacks_by_timeframe": {
            "1m": 90,
            "5m": 60,
            "15m": 48,
            "1h": 24,
            "4h": 16,
            "1d": 10,
        }
    }

    config = validate_config(payload)

    assert config.preprocessing is not None
    assert config.preprocessing.lookbacks_by_timeframe is not None
    assert config.preprocessing.lookbacks_by_timeframe["1d"] == 10


def test_validate_config_rejects_incomplete_preprocessing_lookbacks() -> None:
    """Per-timeframe lookbacks must cover the full modeled stack."""
    payload = build_valid_config()
    payload["preprocessing"] = {
        "lookbacks_by_timeframe": {
            "1m": 90,
            "5m": 60,
        }
    }

    with pytest.raises(ConfigValidationError, match="preprocessing.lookbacks_by_timeframe"):
        validate_config(payload)
