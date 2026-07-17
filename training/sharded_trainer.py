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

from models.losses import AcceptanceMetrics, HeadLosses, MultiTaskTargets, compute_acceptance_metrics, compute_multitask_loss
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
    """Resolved inputs for a sharded training run."""

    store: ShardedDatasetStore
    labels: pd.DataFrame
    folds: tuple[WalkForwardFold, ...]


def run_training_sharded(
    *,
    config: RuntimeConfig,
    manifest_path: str | Path,
    folds: tuple[WalkForwardFold, ...] | list[WalkForwardFold],
    labels: pd.DataFrame,
    model_factory: Callable[[RuntimeConfig, int, int], TrainingModel] | None = None,
    artifact_root: str | Path | None = None,
    input_root: Path | None = None,
    output_root: Path | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> EvaluationSummary:
    """Run deterministic walk-forward training and evaluation from shards.
    
    Args:
        config: Validated runtime configuration.
        manifest_path: Path to preprocessing manifest.
        folds: Walk-forward folds from `U5`.
        labels: Aligned label DataFrame.
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
    if labels.empty:
        raise ShardedTrainerError("labels must not be empty.")

    store = ShardedDatasetStore(manifest_path)
    if store.total_samples != int(len(labels)):
        raise ShardedTrainerError(
            "Shard dataset size mismatch. "
            f"store.total_samples={store.total_samples} labels_rows={len(labels)}"
        )

    validate_temporal_isolation(fold_list, pd.DatetimeIndex(labels["reference_ts"]))
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
    """Resolve labels + folds for a sharded preprocessing output."""
    store = ShardedDatasetStore(manifest_path)

    # Resolve the aligned label parquet written by preprocessing.
    selection = store.manifest.label_selection
    if not selection or "horizon" not in selection or "threshold" not in selection:
        raise ShardedTrainerError("manifest missing label_selection (horizon/threshold).")
    horizon = int(selection["horizon"])
    threshold = float(selection["threshold"])
    labels_path = store.root / "labels" / f"labels_h{horizon}_t{threshold}.parquet"
    if not labels_path.exists():
        raise ShardedTrainerError(f"Aligned labels parquet not found: {labels_path}")

    labels = pd.read_parquet(labels_path)
    if "reference_ts" not in labels.columns:
        raise ShardedTrainerError("labels parquet missing reference_ts column.")

    # Ensure deterministic order.
    labels["reference_ts"] = pd.to_datetime(labels["reference_ts"], errors="raise")
    if labels["reference_ts"].isna().any():
        raise ShardedTrainerError("labels reference_ts contains NaNs.")
    if not pd.DatetimeIndex(labels["reference_ts"]).is_monotonic_increasing:
        labels = labels.sort_values("reference_ts").reset_index(drop=True)

    from training.folds import build_walk_forward_folds  # local import to keep dependency surface tight

    folds = tuple(build_walk_forward_folds(labels, config))
    return ShardedTrainingRunInputs(store=store, labels=labels, folds=folds)


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
            windows_by_tf, reference_close, targets = store.get_slice(batch_start, batch_end)
            outputs = model(
                _move_windows_to_device(windows_by_tf, model.device),
                reference_close.to(device=model.device, dtype=torch.float32),
            )
            losses = compute_multitask_loss(
                event_prediction=outputs.event_prediction,
                boundary_prediction=outputs.boundary_prediction,
                timing_prediction=outputs.timing_prediction,
                confidence_prediction=outputs.confidence_prediction,
                targets=_move_targets_to_device(targets, model.device),
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

    # Metric aggregations with correct denominators.
    event_sse = 0.0
    boundary_abs_sum = 0.0
    boundary_count = 0
    timing_abs_sum = 0.0
    timing_count = 0

    with torch.no_grad():
        for batch_start in range(int(split.start_index), int(split.end_index), batch_size):
            batch_end = min(batch_start + batch_size, int(split.end_index))
            batch_n = int(batch_end - batch_start)

            windows_by_tf, reference_close, targets = store.get_slice(batch_start, batch_end)
            outputs = model(
                _move_windows_to_device(windows_by_tf, model.device),
                reference_close.to(device=model.device, dtype=torch.float32),
            )
            targets_dev = _move_targets_to_device(targets, model.device)

            losses = compute_multitask_loss(
                event_prediction=outputs.event_prediction,
                boundary_prediction=outputs.boundary_prediction,
                timing_prediction=outputs.timing_prediction,
                confidence_prediction=outputs.confidence_prediction,
                targets=targets_dev,
            )
            loss_sums["total_loss"] += float(losses.total_loss.detach().cpu().item()) * batch_n
            loss_sums["event_loss"] += float(losses.event_loss.detach().cpu().item()) * batch_n
            loss_sums["boundary_loss"] += float(losses.boundary_loss.detach().cpu().item()) * batch_n
            loss_sums["timing_loss"] += float(losses.timing_loss.detach().cpu().item()) * batch_n
            loss_sums["confidence_loss"] += float(losses.confidence_loss.detach().cpu().item()) * batch_n
            loss_sums["regime_loss"] += float(losses.regime_loss.detach().cpu().item()) * batch_n

            # Acceptance metrics aggregation (avoid weighting issues on masked timing).
            event_err = (outputs.event_prediction.probabilities - targets_dev.event_flag) ** 2
            event_sse += float(event_err.detach().sum().cpu().item())

            boundary_pred = torch.stack((outputs.boundary_prediction.future_low, outputs.boundary_prediction.future_high), dim=1)
            boundary_truth = torch.stack((targets_dev.future_low, targets_dev.future_high), dim=1).to(boundary_pred.device)
            boundary_abs_sum += float(torch.abs(boundary_pred - boundary_truth).detach().sum().cpu().item())
            boundary_count += int(batch_n * 2)

            valid_timing = (targets_dev.event_start_offset >= 0.0) & (targets_dev.maturity_offset >= 0.0)
            if bool(valid_timing.any()):
                start_err = torch.abs(outputs.timing_prediction.event_start_offset[valid_timing] - targets_dev.event_start_offset[valid_timing])
                maturity_err = torch.abs(outputs.timing_prediction.maturity_offset[valid_timing] - targets_dev.maturity_offset[valid_timing])
                timing_abs_sum += float((start_err.sum() + maturity_err.sum()).detach().cpu().item())
                timing_count += int(valid_timing.detach().sum().cpu().item()) * 2

    mean_losses = {key: value / total for key, value in loss_sums.items()}
    acceptance = AcceptanceMetrics(
        event_brier=event_sse / total,
        boundary_mae=(boundary_abs_sum / boundary_count) if boundary_count > 0 else float("nan"),
        timing_mae=(timing_abs_sum / timing_count) if timing_count > 0 else None,
        confidence_brier=None,
        regime_cross_entropy=None,
    )

    # Wrap scalar losses into tensors to reuse summarize_split_metrics.
    head_losses = HeadLosses(
        event_loss=torch.tensor(mean_losses["event_loss"]),
        boundary_loss=torch.tensor(mean_losses["boundary_loss"]),
        timing_loss=torch.tensor(mean_losses["timing_loss"]),
        confidence_loss=torch.tensor(mean_losses["confidence_loss"]),
        regime_loss=torch.tensor(mean_losses["regime_loss"]),
        total_loss=torch.tensor(mean_losses["total_loss"]),
    )
    return summarize_split_metrics(sample_count=total, losses=head_losses, acceptance_metrics=acceptance)


def _move_windows_to_device(
    windows_by_timeframe: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if set(windows_by_timeframe.keys()) != set(MODELED_TIMEFRAMES):
        raise TrainerError(f"windows_by_timeframe must contain exactly {MODELED_TIMEFRAMES}.")
    return {timeframe: tensor.to(device=device, dtype=torch.float32) for timeframe, tensor in windows_by_timeframe.items()}


def _move_targets_to_device(targets: MultiTaskTargets, device: torch.device) -> MultiTaskTargets:
    return MultiTaskTargets(
        event_flag=targets.event_flag.to(device),
        future_low=targets.future_low.to(device),
        future_high=targets.future_high.to(device),
        event_start_offset=targets.event_start_offset.to(device),
        maturity_offset=targets.maturity_offset.to(device),
        confidence_target=targets.confidence_target.to(device) if targets.confidence_target is not None else None,
        regime_target=targets.regime_target.to(device) if targets.regime_target is not None else None,
    )

