"""U9 walk-forward training orchestration runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from torch import nn

from models.backbone import Backbone
from models.common import TIMEFRAMES
from models.losses import (
    MultiTaskTargets,
    MultiTaskTargetsUnified,
    compute_acceptance_metrics,
    compute_multitask_loss,
    compute_unified_multitask_loss_with_uncertainty,
)
from models.unified_heads import UnifiedBoundaryHead, UnifiedEventHead, UnifiedTimingHead
from training.checkpointing import save_checkpoint, stable_config_hash
from training.config_schema import RuntimeConfig
from training.eval_contract import FoldSplit, WalkForwardFold, fold_iterator
from training.evaluate import EvaluationSummary, FoldEvaluationSummary, run_evaluation
from training.folds import validate_temporal_isolation
from training.metrics import StructuredMetricRecord, append_metric_record, summarize_split_metrics

LOGGER = logging.getLogger(__name__)


class TrainerError(ValueError):
    """Raised when the training runner fails."""


@dataclass(frozen=True)
class TrainingDataset:
    """In-memory dataset for walk-forward training."""

    reference_ts: tuple[datetime, ...]
    windows_by_timeframe: dict[str, torch.Tensor]
    reference_close: torch.Tensor
    targets: MultiTaskTargets


@dataclass(frozen=True)
class ModelOutputs:
    """Typed outputs of the training model (unified multi-timeframe)."""

    unified_event: torch.Tensor
    unified_boundary: torch.Tensor
    unified_timing: torch.Tensor


class TrainingModel(nn.Module):
    """Trainable composition of backbone and unified multi-timeframe heads."""

    def __init__(self, *, config: RuntimeConfig, feature_dim: int, max_horizon_bars: int) -> None:
        super().__init__()
        self.backbone = Backbone(
            feature_dim=feature_dim,
            seed=int(config.training.random_seed),
            device_preference=config.training.device_preference,
            allow_nondeterministic=bool(config.training.allow_nondeterministic),
        )
        # Unified heads (always initialized)
        self.unified_event_head = UnifiedEventHead(latent_dim=self.backbone.latent_dim)
        self.unified_boundary_head = UnifiedBoundaryHead(latent_dim=self.backbone.latent_dim)
        self.unified_timing_head = UnifiedTimingHead(
            latent_dim=self.backbone.latent_dim,
            max_horizon_bars=max_horizon_bars
        )

        self.to(self.backbone.device)

    @property
    def device(self) -> torch.device:
        """Return the execution device."""
        return self.backbone.device

    def forward(self, windows_by_timeframe: dict[str, torch.Tensor], reference_close: torch.Tensor) -> ModelOutputs:
        """Run the backbone and unified heads."""
        backbone_output = self.backbone(windows_by_timeframe)

        # Unified outputs
        unified_event = self.unified_event_head(backbone_output.fused_latent)
        unified_boundary = self.unified_boundary_head(backbone_output.fused_latent)
        unified_timing = self.unified_timing_head(backbone_output.fused_latent)

        return ModelOutputs(
            unified_event=unified_event,
            unified_boundary=unified_boundary,
            unified_timing=unified_timing,
        )


def run_training(
    *,
    config: RuntimeConfig,
    folds: tuple[WalkForwardFold, ...] | list[WalkForwardFold],
    dataset: TrainingDataset,
    model_factory: Callable[[RuntimeConfig, int, int], TrainingModel] | None = None,
    artifact_root: str | Path | None = None,
    input_root: Path | None = None,
    output_root: Path | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> EvaluationSummary:
    """Run deterministic walk-forward training and evaluation.

    Args:
        config: Validated runtime configuration.
        folds: Walk-forward folds from `U5`.
        dataset: In-memory dataset aligned to the fold timeline.
        model_factory: Optional dependency-injected model factory.
        artifact_root: Optional artifact root override (deprecated, use output_root).
        input_root: Optional input root for reading preprocessing artifacts.
        output_root: Optional output root for writing training artifacts.

    Returns:
        Full evaluation summary for the run.
    """
    fold_list = tuple(folds)
    if not fold_list:
        raise TrainerError("folds must not be empty.")

    _validate_dataset(dataset)
    validate_temporal_isolation(fold_list, pd.DatetimeIndex(dataset.reference_ts))
    feature_dim = _resolve_feature_dim(dataset.windows_by_timeframe)
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
            "starting_fold_training",
            extra={
                "event": "starting_fold_training",
                "fold_id": int(fold.fold_id),
                "run_id": run_id,
                "train_start": fold.train.start_ts.isoformat(),
                "test_end": fold.test.end_ts.isoformat(),
            },
        )
        model = factory(config, feature_dim, max_horizon_bars)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training.learning_rate))

        train_dataset = _slice_dataset(dataset, fold.train)
        validation_dataset = _slice_dataset(dataset, fold.validation)
        test_dataset = _slice_dataset(dataset, fold.test)

        _train_model(
            model=model,
            optimizer=optimizer,
            dataset=train_dataset,
            batch_size=int(config.training.batch_size),
            max_epochs=int(config.training.max_epochs),
        )

        train_metrics = _evaluate_split(model=model, dataset=train_dataset)
        validation_metrics = _evaluate_split(model=model, dataset=validation_dataset)
        test_metrics = _evaluate_split(model=model, dataset=test_dataset)

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
        for split_name, split_dataset, metrics in (
            ("train", train_dataset, train_metrics),
            ("validation", validation_dataset, validation_metrics),
            ("test", test_dataset, test_metrics),
        ):
            append_metric_record(
                metrics_log_path,
                StructuredMetricRecord(
                    run_id=run_id,
                    fold_id=int(fold.fold_id),
                    split_name=split_name,
                    sample_count=len(split_dataset.reference_ts),
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


def _default_model_factory(config: RuntimeConfig, feature_dim: int, max_horizon_bars: int) -> TrainingModel:
    return TrainingModel(config=config, feature_dim=feature_dim, max_horizon_bars=max_horizon_bars)


def _train_model(
    *,
    model: TrainingModel,
    optimizer: torch.optim.Optimizer,
    dataset: TrainingDataset,
    batch_size: int,
    max_epochs: int,
) -> None:
    if batch_size <= 0:
        raise TrainerError("batch_size must be positive.")
    if max_epochs <= 0:
        raise TrainerError("max_epochs must be positive.")

    sample_count = len(dataset.reference_ts)
    for _epoch in range(max_epochs):
        model.train()
        for batch_start in range(0, sample_count, batch_size):
            batch_end = min(batch_start + batch_size, sample_count)
            batch_dataset = _slice_dataset(dataset, FoldSplit(batch_start, batch_end, dataset.reference_ts[batch_start], dataset.reference_ts[batch_end - 1]))
            outputs = model(
                _move_windows_to_device(batch_dataset.windows_by_timeframe, model.device),
                batch_dataset.reference_close.to(model.device),
            )
            losses = compute_unified_multitask_loss_with_uncertainty(
                unified_event=outputs.unified_event,
                unified_boundary=outputs.unified_boundary,
                unified_timing=outputs.unified_timing,
                event_log_sigma=model.unified_event_head.log_sigma,
                boundary_log_sigma=model.unified_boundary_head.log_sigma,
                timing_log_sigma=model.unified_timing_head.log_sigma,
                targets=_move_targets_unified_to_device(batch_dataset.targets, model.device),
            )
            optimizer.zero_grad()
            losses.total_loss.backward()
            optimizer.step()


def _evaluate_split(*, model: TrainingModel, dataset: TrainingDataset) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        outputs = model(
            _move_windows_to_device(dataset.windows_by_timeframe, model.device),
            dataset.reference_close.to(model.device),
        )
        targets = _move_targets_unified_to_device(dataset.targets, model.device)
        losses = compute_unified_multitask_loss_with_uncertainty(
            unified_event=outputs.unified_event,
            unified_boundary=outputs.unified_boundary,
            unified_timing=outputs.unified_timing,
            event_log_sigma=model.unified_event_head.log_sigma,
            boundary_log_sigma=model.unified_boundary_head.log_sigma,
            timing_log_sigma=model.unified_timing_head.log_sigma,
            targets=targets,
        )
        # For now, use simple metrics - will be updated in Phase 6
        # Convert UncertaintyWeightedLosses to HeadLosses for summarize_split_metrics
        from models.losses import HeadLosses
        head_losses = HeadLosses(
            event_loss=losses.event_loss,
            boundary_loss=losses.boundary_loss,
            timing_loss=losses.timing_loss,
            confidence_loss=torch.tensor(0.0, device=model.device),
            regime_loss=torch.tensor(0.0, device=model.device),
            total_loss=losses.total_loss,
        )
        # Placeholder acceptance metrics - will be updated in Phase 6
        from models.losses import AcceptanceMetrics
        acceptance_metrics = AcceptanceMetrics(
            event_brier=float(losses.event_loss.detach().cpu().item()),
            boundary_mae=float(losses.boundary_loss.detach().cpu().item()),
            timing_mae=float(losses.timing_loss.detach().cpu().item()),
            confidence_brier=None,
            regime_cross_entropy=None,
        )
    return summarize_split_metrics(
        sample_count=len(dataset.reference_ts),
        losses=head_losses,
        acceptance_metrics=acceptance_metrics,
    )


def _validate_dataset(dataset: TrainingDataset) -> None:
    if set(dataset.windows_by_timeframe.keys()) != set(TIMEFRAMES):
        raise TrainerError(f"dataset.windows_by_timeframe must contain exactly {TIMEFRAMES}.")
    if not dataset.reference_ts:
        raise TrainerError("dataset.reference_ts must not be empty.")
    if dataset.reference_close.ndim != 1:
        raise TrainerError("dataset.reference_close must have shape (batch,).")
    if dataset.reference_close.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise TrainerError("dataset.reference_close must have a floating-point dtype.")

    expected = len(dataset.reference_ts)
    if dataset.reference_close.shape[0] != expected:
        raise TrainerError("reference_close length must match reference_ts length.")
    for timeframe, windows in dataset.windows_by_timeframe.items():
        if windows.ndim != 3:
            raise TrainerError(f"windows for timeframe={timeframe} must have shape (batch, lookback, feature_dim).")
        if windows.shape[0] != expected:
            raise TrainerError(f"windows batch size must match reference_ts for timeframe={timeframe}.")
    
    # Validate unified targets
    if isinstance(dataset.targets, MultiTaskTargetsUnified):
        for timeframe in TIMEFRAMES:
            if timeframe not in dataset.targets.event_flag:
                raise TrainerError(f"targets.event_flag missing timeframe={timeframe}")
            for field_name in ("event_flag", "future_low", "future_high", "event_start_offset", "maturity_offset"):
                tensor = getattr(dataset.targets, field_name)[timeframe]
                if tensor.ndim != 2 or tensor.shape[0] != expected:
                    raise TrainerError(f"targets.{field_name}[{timeframe}] must have shape (batch, 3) aligned to reference_ts.")
    else:
        # Legacy MultiTaskTargets validation
        for field_name in ("event_flag", "future_low", "future_high", "event_start_offset", "maturity_offset"):
            tensor = getattr(dataset.targets, field_name)
            if tensor.ndim != 1 or tensor.shape[0] != expected:
                raise TrainerError(f"targets.{field_name} must have shape (batch,) aligned to reference_ts.")
        if dataset.targets.confidence_target is not None and dataset.targets.confidence_target.shape[0] != expected:
            raise TrainerError("targets.confidence_target must align to reference_ts when provided.")


def _resolve_feature_dim(windows_by_timeframe: dict[str, torch.Tensor]) -> int:
    feature_dim = int(windows_by_timeframe["1m"].shape[2])
    for timeframe, windows in windows_by_timeframe.items():
        if int(windows.shape[2]) != feature_dim:
            raise TrainerError(f"feature_dim mismatch for timeframe={timeframe}.")
    return feature_dim


def _slice_dataset(dataset: TrainingDataset, split: FoldSplit) -> TrainingDataset:
    start = int(split.start_index)
    end = int(split.end_index)
    
    # Slice targets based on type
    if isinstance(dataset.targets, MultiTaskTargetsUnified):
        sliced_targets = MultiTaskTargetsUnified(
            event_flag={tf: dataset.targets.event_flag[tf][start:end] for tf in TIMEFRAMES},
            future_low={tf: dataset.targets.future_low[tf][start:end] for tf in TIMEFRAMES},
            future_high={tf: dataset.targets.future_high[tf][start:end] for tf in TIMEFRAMES},
            event_start_offset={tf: dataset.targets.event_start_offset[tf][start:end] for tf in TIMEFRAMES},
            maturity_offset={tf: dataset.targets.maturity_offset[tf][start:end] for tf in TIMEFRAMES},
        )
    else:
        sliced_targets = MultiTaskTargets(
            event_flag=dataset.targets.event_flag[start:end],
            future_low=dataset.targets.future_low[start:end],
            future_high=dataset.targets.future_high[start:end],
            event_start_offset=dataset.targets.event_start_offset[start:end],
            maturity_offset=dataset.targets.maturity_offset[start:end],
            confidence_target=(
                dataset.targets.confidence_target[start:end] if dataset.targets.confidence_target is not None else None
            ),
            regime_target=dataset.targets.regime_target[start:end] if dataset.targets.regime_target is not None else None,
        )
    
    return TrainingDataset(
        reference_ts=dataset.reference_ts[start:end],
        windows_by_timeframe={timeframe: windows[start:end] for timeframe, windows in dataset.windows_by_timeframe.items()},
        reference_close=dataset.reference_close[start:end],
        targets=sliced_targets,
    )


def _move_windows_to_device(
    windows_by_timeframe: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
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


def _move_targets_unified_to_device(targets: MultiTaskTargetsUnified, device: torch.device) -> MultiTaskTargetsUnified:
    return MultiTaskTargetsUnified(
        event_flag={tf: targets.event_flag[tf].to(device) for tf in TIMEFRAMES},
        future_low={tf: targets.future_low[tf].to(device) for tf in TIMEFRAMES},
        future_high={tf: targets.future_high[tf].to(device) for tf in TIMEFRAMES},
        event_start_offset={tf: targets.event_start_offset[tf].to(device) for tf in TIMEFRAMES},
        maturity_offset={tf: targets.maturity_offset[tf].to(device) for tf in TIMEFRAMES},
    )
