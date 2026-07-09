"""Unit tests for U9 checkpointing."""

from __future__ import annotations

from pathlib import Path

import torch

from training.checkpointing import load_checkpoint, save_checkpoint
from training.config_loader import validate_config


def _build_config(output_dir: Path) -> object:
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
            "features": {"enabled_features": ["returns"], "deterministic_derived_features": ["returns"]},
            "labeling": {"thresholds": [10.0], "horizon_bars": [5]},
            "walk_forward": {"train_bars": 4, "validation_bars": 2, "test_bars": 2, "step_bars": 2},
            "training": {
                "random_seed": 7,
                "batch_size": 2,
                "learning_rate": 0.001,
                "max_epochs": 1,
                "device_preference": "cpu",
                "allow_nondeterministic": False,
            },
            "retrieval": {"top_k_analogs": 2},
            "api": {"host": "127.0.0.1", "port": 8000},
            "reporting": {"output_dir": str(output_dir)},
            "schedules": {"reporting_cron": "0 9 * * *", "retraining_review_cron": "0 0 * * 1"},
        }
    )


def test_u9_checkpoint_save_and_load_are_reproducible(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = {"total_loss": 0.25, "event_brier": 0.10}

    first = save_checkpoint(
        artifact_root=tmp_path,
        config=config,
        fold_id=1,
        model=model,
        optimizer=optimizer,
        metrics=metrics,
    )
    second = save_checkpoint(
        artifact_root=tmp_path,
        config=config,
        fold_id=1,
        model=model,
        optimizer=optimizer,
        metrics=metrics,
    )
    loaded = load_checkpoint(first.checkpoint_path)

    assert first.checkpoint_path == second.checkpoint_path
    assert first.config_hash == second.config_hash
    assert loaded.model_id == first.model_id
    assert loaded.fold_id == 1
    assert loaded.metrics == metrics
    assert first.checkpoint_path.exists()
