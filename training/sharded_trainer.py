"""Sharded training runner (Kaggle-scale).

This module mirrors `training.trainer.run_training()` but consumes a disk-backed
sharded dataset produced by `Preprocessing.py`.

It is intentionally added as a parallel workflow to avoid breaking the existing
bundle-based CLI and tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from models.losses import (
    AcceptanceMetrics,
    HeadLosses,
    MultiTaskTargets,
    MultiTaskTargetsUnified,
    compute_acceptance_metrics,
    compute_multitask_loss,
    compute_unified_multitask_loss_with_uncertainty,
)
from training.checkpointing import save_checkpoint, stable_config_hash
from training.config_schema import RuntimeConfig
from training.eval_contract import FoldSplit, WalkForwardFold, fold_iterator
from training.evaluate import EvaluationSummary, FoldEvaluationSummary, run_evaluation
from training.folds import validate_temporal_isolation
from training.metrics import StructuredMetricRecord, append_metric_record, summarize_split_metrics
from training.sharded_store import MODELED_TIMEFRAMES, ShardedDatasetStore, ShardedStoreError
from training.trainer import TrainerError, TrainingModel

LOGGER = logging.getLogger(__name__)


class ShardedTrainerError(ValueError):
    """Raised when sharded training fails."""


@dataclass(frozen=True)
class ShardedTrainingRunInputs:
    """Resolved inputs for a sharded training run (unified multi-timeframe)."""

    store: ShardedDatasetStore
    folds: tuple[WalkForwardFold, ...]


def run_training_sharded(
    *,
    config: RuntimeConfig,
    manifest_path: str | Path,
    folds: tuple[WalkForwardFold, ...] | list[WalkForwardFold],
    model_factory: Callable[[RuntimeConfig, int, int], TrainingModel] | None = None,
    artifact_root: str | Path | None = None,
    input_root: Path | None = None,
    output_root: Path | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> EvaluationSummary:
    """Run deterministic walk-forward training and evaluation from shards (unified multi-timeframe).
    
    Args:
        config: Validated runtime configuration.
        manifest_path: Path to preprocessing manifest.
        folds: Walk-forward folds from `U5`.
        model_factory: Optional dependency-injected model factory.
        artifact_root: Optional artifact root override (deprecated, use output_root).
        input_root: Optional input root for reading preprocessing artifacts.
        output_root: Optional output root for writing training artifacts.
        checkpoint_metadata: Optional checkpoint metadata.
        
    Returns:
        Full evaluation summary for the run.
    """
    fold_list = tuple(folds)
    if not fold_list:
        raise ShardedTrainerError("folds must not be empty.")

    store = ShardedDatasetStore(manifest_path, config=config, debug=True)

    # Use reference timestamps from manifest for fold validation
    reference_ts = store.get_reference_timestamps()
    validate_temporal_isolation(fold_list, pd.DatetimeIndex(reference_ts))
    feature_dim = int(store.feature_dim)
    max_horizon_bars = int(max(config.labeling.horizon_bars))
    
    # Resolve output root: output_root > artifact_root > config.training.output_root > config.reporting.output_dir
    if artifact_root is not None:
        import warnings
        warnings.warn(
            "artifact_root parameter is deprecated. Use output_root instead.",
            DeprecationWarning,
            stacklevel=2
        )
        resolved_root = Path(artifact_root)
    elif output_root is not None:
        resolved_root = Path(output_root)
    elif config.training.output_root is not None:
        resolved_root = config.training.output_root
    else:
        resolved_root = Path(config.reporting.output_dir)
    
    run_id = stable_config_hash(config)
    factory = model_factory or _default_model_factory

    fold_summaries: list[FoldEvaluationSummary] = []
    for fold in fold_iterator(fold_list):
        LOGGER.info(
            "starting_fold_training_sharded",
            extra={
                "event": "starting_fold_training_sharded",
                "fold_id": int(fold.fold_id),
                "run_id": run_id,
                "train_start": fold.train.start_ts.isoformat(),
                "test_end": fold.test.end_ts.isoformat(),
            },
        )

        model = factory(config, feature_dim, max_horizon_bars)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training.learning_rate))

        _train_model_sharded(
            model=model,
            optimizer=optimizer,
            store=store,
            split=fold.train,
            batch_size=int(config.training.batch_size),
            max_epochs=int(config.training.max_epochs),
        )

        train_metrics = _evaluate_split_sharded(model=model, store=store, split=fold.train)
        validation_metrics = _evaluate_split_sharded(model=model, store=store, split=fold.validation)
        test_metrics = _evaluate_split_sharded(model=model, store=store, split=fold.test)

        checkpoint = save_checkpoint(
            artifact_root=resolved_root,
            config=config,
            fold_id=int(fold.fold_id),
            model=model,
            optimizer=optimizer,
            metrics=validation_metrics,
            metadata=checkpoint_metadata,
        )
        metrics_log_path = checkpoint.artifact_dir / "metrics.jsonl"
        if metrics_log_path.exists():
            metrics_log_path.unlink()
        for split_name, split, metrics in (
            ("train", fold.train, train_metrics),
            ("validation", fold.validation, validation_metrics),
            ("test", fold.test, test_metrics),
        ):
            append_metric_record(
                metrics_log_path,
                StructuredMetricRecord(
                    run_id=run_id,
                    fold_id=int(fold.fold_id),
                    split_name=split_name,
                    sample_count=int(split.size),
                    metrics=metrics,
                ),
            )

        fold_summaries.append(
            FoldEvaluationSummary(
                fold_id=int(fold.fold_id),
                model_id=checkpoint.model_id,
                config_hash=checkpoint.config_hash,
                checkpoint_path=checkpoint.checkpoint_path,
                artifact_dir=checkpoint.artifact_dir,
                metrics_log_path=metrics_log_path,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                train_range=(fold.train.start_ts, fold.train.end_ts),
                validation_range=(fold.validation.start_ts, fold.validation.end_ts),
                test_range=(fold.test.start_ts, fold.test.end_ts),
            )
        )

    return run_evaluation(config=config, fold_summaries=tuple(fold_summaries))


def resolve_sharded_inputs(
    *,
    config: RuntimeConfig,
    manifest_path: str | Path,
) -> ShardedTrainingRunInputs:
    """Resolve inputs for a sharded preprocessing output (unified multi-timeframe)."""
    store = ShardedDatasetStore(manifest_path, config=config, debug=True)

    # Build folds from reference timestamps in the manifest
    reference_ts = store.get_reference_timestamps()
    from training.folds import build_walk_forward_folds
    folds = build_walk_forward_folds(
        pd.DataFrame({"reference_ts": reference_ts}),
        config
    )

    return ShardedTrainingRunInputs(store=store, folds=tuple(folds))


def _default_model_factory(config: RuntimeConfig, feature_dim: int, max_horizon_bars: int) -> TrainingModel:
    return TrainingModel(config=config, feature_dim=feature_dim, max_horizon_bars=max_horizon_bars)


def _train_model_sharded(
    *,
    model: TrainingModel,
    optimizer: torch.optim.Optimizer,
    store: ShardedDatasetStore,
    split: FoldSplit,
    batch_size: int,
    max_epochs: int,
) -> None:
    if batch_size <= 0:
        raise ShardedTrainerError("batch_size must be positive.")
    if max_epochs <= 0:
        raise ShardedTrainerError("max_epochs must be positive.")
    if split.size <= 0:
        raise ShardedTrainerError("split must be non-empty.")

    for _epoch in range(max_epochs):
        model.train()
        for batch_start in range(int(split.start_index), int(split.end_index), batch_size):
            batch_end = min(batch_start + batch_size, int(split.end_index))
            windows_by_tf, reference_close, targets = store.get_slice_unified(batch_start, batch_end)
            outputs = model(
                _move_windows_to_device(windows_by_tf, model.device),
                reference_close.to(device=model.device, dtype=torch.float32),
            )
            losses = compute_unified_multitask_loss_with_uncertainty(
                unified_event=outputs.unified_event,
                unified_boundary=outputs.unified_boundary,
                unified_timing=outputs.unified_timing,
                event_log_sigma=model.unified_event_head.log_sigma,
                boundary_log_sigma=model.unified_boundary_head.log_sigma,
                timing_log_sigma=model.unified_timing_head.log_sigma,
                targets=_move_targets_unified_to_device(targets, model.device),
            )
            optimizer.zero_grad()
            losses.total_loss.backward()
            optimizer.step()


def _evaluate_split_sharded(*, model: TrainingModel, store: ShardedDatasetStore, split: FoldSplit) -> dict[str, float]:
    if split.size <= 0:
        raise ShardedTrainerError("split must be non-empty.")

    model.eval()
    total = int(split.size)
    batch_size = min(4096, total)

    # Weighted aggregations for mean losses.
    loss_sums = {
        "total_loss": 0.0,
        "event_loss": 0.0,
        "boundary_loss": 0.0,
        "timing_loss": 0.0,
        "confidence_loss": 0.0,
        "regime_loss": 0.0,
    }

    with torch.no_grad():
        for batch_start in range(int(split.start_index), int(split.end_index), batch_size):
            batch_end = min(batch_start + batch_size, int(split.end_index))
            batch_n = int(batch_end - batch_start)

            windows_by_tf, reference_close, targets = store.get_slice_unified(batch_start, batch_end)
            outputs = model(
                _move_windows_to_device(windows_by_tf, model.device),
                reference_close.to(device=model.device, dtype=torch.float32),
            )
            targets_dev = _move_targets_unified_to_device(targets, model.device)

            losses = compute_unified_multitask_loss_with_uncertainty(
                unified_event=outputs.unified_event,
                unified_boundary=outputs.unified_boundary,
                unified_timing=outputs.unified_timing,
                event_log_sigma=model.unified_event_head.log_sigma,
                boundary_log_sigma=model.unified_boundary_head.log_sigma,
                timing_log_sigma=model.unified_timing_head.log_sigma,
                targets=targets_dev,
            )
            loss_sums["total_loss"] += float(losses.total_loss.detach().cpu().item()) * batch_n
            loss_sums["event_loss"] += float(losses.event_loss.detach().cpu().item()) * batch_n
            loss_sums["boundary_loss"] += float(losses.boundary_loss.detach().cpu().item()) * batch_n
            loss_sums["timing_loss"] += float(losses.timing_loss.detach().cpu().item()) * batch_n
            loss_sums["confidence_loss"] = 0.0  # No confidence head in unified
            loss_sums["regime_loss"] = 0.0  # No regime head in unified

    # Compute mean losses
    metrics = {key: value / total for key, value in loss_sums.items()}
    
    # Placeholder acceptance metrics - will be updated in Phase 6
    metrics["event_brier"] = metrics["event_loss"]
    metrics["boundary_mae"] = metrics["boundary_loss"]
    metrics["timing_mae"] = metrics["timing_loss"]
    metrics["confidence_brier"] = -1.0
    metrics["regime_cross_entropy"] = -1.0
    
    return metrics


def _move_windows_to_device(
    windows_by_timeframe: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if set(windows_by_timeframe.keys()) != set(MODELED_TIMEFRAMES):
        raise TrainerError(f"windows_by_timeframe must contain exactly {MODELED_TIMEFRAMES}.")
    return {timeframe: tensor.to(device=device, dtype=torch.float32) for timeframe, tensor in windows_by_timeframe.items()}


def _move_targets_unified_to_device(targets: MultiTaskTargetsUnified, device: torch.device) -> MultiTaskTargetsUnified:
    return MultiTaskTargetsUnified(
        event_flag={tf: targets.event_flag[tf].to(device) for tf in MODELED_TIMEFRAMES},
        future_low={tf: targets.future_low[tf].to(device) for tf in MODELED_TIMEFRAMES},
        future_high={tf: targets.future_high[tf].to(device) for tf in MODELED_TIMEFRAMES},
        event_start_offset={tf: targets.event_start_offset[tf].to(device) for tf in MODELED_TIMEFRAMES},
        maturity_offset={tf: targets.maturity_offset[tf].to(device) for tf in MODELED_TIMEFRAMES},
    )

