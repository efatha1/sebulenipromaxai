"""Unit tests for U1 config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from training.config_loader import (
    ConfigFileNotFoundError,
    ConfigOverrideError,
    load_config,
)


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


def write_yaml(path: Path, payload: dict[str, object]) -> None:
    """Write YAML test data."""
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_load_config_applies_environment_overrides(tmp_path: Path) -> None:
    """Environment overrides should deterministically update known paths."""
    config_path = tmp_path / "config.yaml"
    write_yaml(config_path, build_valid_config())

    config = load_config(
        config_path,
        env={
            "SEBULENI__INSTRUMENT__INSTRUMENT_ID": "OVERRIDDEN_INSTRUMENT",
            "SEBULENI__TRAINING__RANDOM_SEED": "7",
            "SEBULENI__RETRIEVAL__TOP_K_ANALOGS": "9",
        },
    )

    assert config.instrument.instrument_id == "OVERRIDDEN_INSTRUMENT"
    assert config.training.random_seed == 7
    assert config.retrieval.top_k_analogs == 9


def test_load_config_rejects_unknown_override_path(tmp_path: Path) -> None:
    """Unknown override keys should fail early."""
    config_path = tmp_path / "config.yaml"
    write_yaml(config_path, build_valid_config())

    with pytest.raises(ConfigOverrideError, match="UNKNOWN_FIELD"):
        load_config(
            config_path,
            env={"SEBULENI__INSTRUMENT__UNKNOWN_FIELD": "bad"},
        )


def test_load_config_missing_file_fails() -> None:
    """Missing config files should fail early."""
    with pytest.raises(ConfigFileNotFoundError):
        load_config("missing-config.yaml")

