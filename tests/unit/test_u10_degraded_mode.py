"""Unit tests for U10 degraded-mode advisory behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from inference.model_store import ActiveModelArtifacts, load_active_model
from inference.predictor import predict
from inference.runtime_features import build_runtime_window
from models.explanation import RetrievalMemoryRecord
from training.checkpointing import save_checkpoint
from training.config_loader import validate_config
from training.contracts import PredictionRequestContract
from training.trainer import TrainingModel


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
            "features": {
                "enabled_features": ["returns", "calendar_time"],
                "deterministic_derived_features": ["returns", "calendar_time"],
            },
            "labeling": {"thresholds": [10.0], "horizon_bars": [3, 5]},
            "walk_forward": {"train_bars": 4, "validation_bars": 2, "test_bars": 2, "step_bars": 2},
            "training": {
                "random_seed": 7,
                "batch_size": 2,
                "learning_rate": 0.001,
                "max_epochs": 1,
                "device_preference": "cpu",
                "allow_nondeterministic": False,
            },
            "retrieval": {"top_k_analogs": 3},
            "api": {"host": "127.0.0.1", "port": 8000},
            "reporting": {"output_dir": str(output_dir)},
            "schedules": {"reporting_cron": "0 9 * * *", "retraining_review_cron": "0 0 * * 1"},
        }
    )


def _build_request() -> PredictionRequestContract:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    bars = [
        {
            "timestamp": (start + timedelta(minutes=index)).isoformat(),
            "open": 100.0 + index,
            "high": 100.5 + index,
            "low": 99.5 + index,
            "close": 100.25 + index,
        }
        for index in range(6)
    ]
    return PredictionRequestContract.model_validate(
        {
            "instrument_id": "TEST_INSTRUMENT",
            "bars_1m": bars,
            "horizon_mode": "single",
            "horizon_bars": 3,
            "threshold": 10.0,
            "top_k_analogs": 3,
        }
    )


def test_u10_predict_emits_low_confidence_advisory_when_support_is_weak(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    request = _build_request()
    lookbacks = {timeframe: 1 for timeframe in ("1m", "5m", "15m", "1h", "4h", "1d")}
    runtime_window = build_runtime_window(
        request,
        config,
        lookbacks,
        current_time=request.bars_1m[-1].timestamp + timedelta(seconds=30),
    )
    feature_dim = int(runtime_window.windows_by_timeframe["1m"].shape[2])

    model = TrainingModel(config=config, feature_dim=feature_dim, max_horizon_bars=5)
    optimizer = None
    checkpoint = save_checkpoint(
        artifact_root=tmp_path,
        config=config,
        fold_id=0,
        model=model,
        optimizer=optimizer,
        metrics={"total_loss": 1.0},
        metadata={"lookbacks_by_timeframe": dict(lookbacks)},
    )
    retrieval_memory = (
        RetrievalMemoryRecord(
            analog_id="fold-0:2026-01-01T00:03:00+00:00",
            reference_ts=datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc),
            latent_vector=tuple(0.0 for _ in range(feature_dim)),
            outcome_summary="event_observed=1; future_low=98.0000; future_high=102.0000; duration_bars=2.0",
            event_observed=1.0,
            future_low=98.0,
            future_high=102.0,
            duration_bars=2.0,
            source_split="train",
            source_fold_id="fold-0",
        ),
    )
    active_model = load_active_model(
        config,
        ActiveModelArtifacts(
            checkpoint_path=checkpoint.checkpoint_path,
            lookbacks_by_timeframe=lookbacks,
            retrieval_memory=retrieval_memory,
        ),
    )

    responses = predict(
        request,
        config,
        active_model,
        current_time=request.bars_1m[-1].timestamp + timedelta(seconds=30),
    )

    assert len(responses) == 1
    response = responses[0]
    assert response.prediction.low_confidence_advisory is True
    assert "limited" in response.grounded_natural_language_explanation.lower()
