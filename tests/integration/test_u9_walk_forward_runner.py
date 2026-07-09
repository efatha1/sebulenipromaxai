"""Integration tests for the U9 walk-forward training runner."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from models.common import TIMEFRAMES
from models.losses import MultiTaskTargets
from training.config_loader import validate_config
from training.folds import build_walk_forward_folds
from training.trainer import TrainingDataset, run_training


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


def test_u9_walk_forward_runner_emits_artifacts_and_metrics(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    reference_ts = [datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc) for i in range(22)]
    labeled_df = pd.DataFrame({"reference_ts": reference_ts, "ambiguous": [False] * len(reference_ts)})
    folds = build_walk_forward_folds(labeled_df, config)

    torch.manual_seed(5)
    windows_by_timeframe = {
        timeframe: torch.randn(len(reference_ts), 3, 4, dtype=torch.float32)
        for timeframe in TIMEFRAMES
    }
    dataset = TrainingDataset(
        reference_ts=tuple(reference_ts),
        windows_by_timeframe=windows_by_timeframe,
        reference_close=torch.linspace(100.0, 101.1, steps=len(reference_ts)),
        targets=MultiTaskTargets(
            event_flag=torch.tensor([0.0, 1.0] * 11),
            future_low=torch.linspace(98.0, 99.1, steps=len(reference_ts)),
            future_high=torch.linspace(101.0, 103.2, steps=len(reference_ts)),
            event_start_offset=torch.tensor([-1.0, 2.0] * 11),
            maturity_offset=torch.tensor([-1.0, 4.0] * 11),
            confidence_target=torch.tensor([0.1, 0.9] * 11),
            regime_target=None,
        ),
    )

    result = run_training(config=config, folds=folds, dataset=dataset, artifact_root=tmp_path)

    assert len(result.fold_summaries) == 3
    assert result.aggregate_metrics["fold_count"] == 3.0
    assert result.candidate_model.artifact_path.exists()
    for summary in result.fold_summaries:
        assert summary.checkpoint_path.exists()
        assert summary.metrics_log_path.exists()
        assert "total_loss" in summary.validation_metrics


def test_u9_walk_forward_runner_overwrites_metrics_artifacts_reproducibly(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    reference_ts = [datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc) for i in range(22)]
    labeled_df = pd.DataFrame({"reference_ts": reference_ts, "ambiguous": [False] * len(reference_ts)})
    folds = build_walk_forward_folds(labeled_df, config)

    torch.manual_seed(5)
    windows_by_timeframe = {
        timeframe: torch.randn(len(reference_ts), 3, 4, dtype=torch.float32)
        for timeframe in TIMEFRAMES
    }
    dataset = TrainingDataset(
        reference_ts=tuple(reference_ts),
        windows_by_timeframe=windows_by_timeframe,
        reference_close=torch.linspace(100.0, 101.1, steps=len(reference_ts)),
        targets=MultiTaskTargets(
            event_flag=torch.tensor([0.0, 1.0] * 11),
            future_low=torch.linspace(98.0, 99.1, steps=len(reference_ts)),
            future_high=torch.linspace(101.0, 103.2, steps=len(reference_ts)),
            event_start_offset=torch.tensor([-1.0, 2.0] * 11),
            maturity_offset=torch.tensor([-1.0, 4.0] * 11),
            confidence_target=torch.tensor([0.1, 0.9] * 11),
            regime_target=None,
        ),
    )

    first = run_training(config=config, folds=folds, dataset=dataset, artifact_root=tmp_path)
    second = run_training(config=config, folds=folds, dataset=dataset, artifact_root=tmp_path)

    assert first.candidate_model.artifact_path == second.candidate_model.artifact_path
    for summary in second.fold_summaries:
        lines = summary.metrics_log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
