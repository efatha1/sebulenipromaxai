"""U7 multi-task losses and evaluation hooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from models.boundary_head import BoundaryPrediction
from models.confidence_head import ConfidencePrediction
from models.event_head import EventPrediction
from models.regime_head import RegimePrediction
from models.timing_head import TimingPrediction


class LossError(ValueError):
    """Raised when loss inputs are invalid."""


@dataclass(frozen=True)
class LossWeights:
    """Configurable task weights for the U7 multi-task objective."""

    event: float = 1.0
    boundary: float = 1.0
    timing: float = 1.0
    confidence: float = 1.0
    regime: float = 1.0


@dataclass(frozen=True)
class MultiTaskTargets:
    """Typed training targets aligned with the approved U4 semantics."""

    event_flag: torch.Tensor
    future_low: torch.Tensor
    future_high: torch.Tensor
    event_start_offset: torch.Tensor
    maturity_offset: torch.Tensor
    confidence_target: torch.Tensor | None = None
    regime_target: torch.Tensor | None = None


@dataclass(frozen=True)
class HeadLosses:
    """Typed per-task and aggregate loss outputs."""

    event_loss: torch.Tensor
    boundary_loss: torch.Tensor
    timing_loss: torch.Tensor
    confidence_loss: torch.Tensor
    regime_loss: torch.Tensor
    total_loss: torch.Tensor


@dataclass(frozen=True)
class AcceptanceMetrics:
    """Typed evaluation metrics for downstream offline reporting."""

    event_brier: float
    boundary_mae: float
    timing_mae: float | None
    confidence_brier: float | None
    regime_cross_entropy: float | None


def compute_event_loss(prediction: EventPrediction, target: torch.Tensor) -> torch.Tensor:
    """Compute binary event loss from logits and event flags."""
    target_float = _validate_probability_target(target=target, field_name="event_flag")
    _validate_shape(prediction.logits, target_float, "event logits")
    return F.binary_cross_entropy_with_logits(prediction.logits, target_float)


def compute_boundary_loss(
    prediction: BoundaryPrediction,
    *,
    future_low: torch.Tensor,
    future_high: torch.Tensor,
) -> torch.Tensor:
    """Compute the future-boundary regression loss."""
    low = _validate_float_vector(target=future_low, field_name="future_low")
    high = _validate_float_vector(target=future_high, field_name="future_high")
    _validate_shape(prediction.future_low, low, "future_low")
    _validate_shape(prediction.future_high, high, "future_high")

    pred = torch.stack((prediction.future_low, prediction.future_high), dim=1)
    truth = torch.stack((low, high), dim=1).to(device=pred.device, dtype=pred.dtype)
    return F.smooth_l1_loss(pred, truth)


def compute_timing_loss(
    prediction: TimingPrediction,
    *,
    event_start_offset: torch.Tensor,
    maturity_offset: torch.Tensor,
) -> torch.Tensor:
    """Compute timing loss on valid timing targets only."""
    start = _validate_float_vector(target=event_start_offset, field_name="event_start_offset")
    maturity = _validate_float_vector(target=maturity_offset, field_name="maturity_offset")
    _validate_shape(prediction.event_start_offset, start, "event_start_offset")
    _validate_shape(prediction.maturity_offset, maturity, "maturity_offset")

    valid_start = start >= 0.0
    valid_maturity = maturity >= 0.0
    start_loss = _masked_smooth_l1(prediction.event_start_offset, start, valid_start)
    maturity_loss = _masked_smooth_l1(prediction.maturity_offset, maturity, valid_maturity)
    return 0.5 * (start_loss + maturity_loss)


def compute_confidence_loss(prediction: ConfidencePrediction, target: torch.Tensor) -> torch.Tensor:
    """Compute confidence calibration loss."""
    target_float = _validate_probability_target(target=target, field_name="confidence_target")
    _validate_shape(prediction.confidence, target_float, "confidence")
    return F.mse_loss(prediction.confidence, target_float.to(prediction.confidence.device))


def compute_regime_loss(prediction: RegimePrediction, target: torch.Tensor) -> torch.Tensor:
    """Compute categorical regime loss."""
    target_index = _validate_class_target(target=target, field_name="regime_target")
    if prediction.logits.ndim != 2:
        raise LossError("regime logits must have shape (batch, num_regimes).")
    if prediction.logits.shape[0] != target_index.shape[0]:
        raise LossError("regime logits batch size must match regime_target batch size.")
    return F.cross_entropy(prediction.logits, target_index.to(prediction.logits.device))


def compute_multitask_loss(
    *,
    event_prediction: EventPrediction,
    boundary_prediction: BoundaryPrediction,
    timing_prediction: TimingPrediction,
    confidence_prediction: ConfidencePrediction,
    targets: MultiTaskTargets,
    regime_prediction: RegimePrediction | None = None,
    weights: LossWeights | None = None,
) -> HeadLosses:
    """Compute the aggregate U7 objective with optional confidence/regime terms."""
    loss_weights = weights or LossWeights()
    _validate_loss_weights(loss_weights)

    event_loss = compute_event_loss(event_prediction, targets.event_flag)
    boundary_loss = compute_boundary_loss(
        boundary_prediction,
        future_low=targets.future_low,
        future_high=targets.future_high,
    )
    timing_loss = compute_timing_loss(
        timing_prediction,
        event_start_offset=targets.event_start_offset,
        maturity_offset=targets.maturity_offset,
    )
    confidence_loss = (
        compute_confidence_loss(confidence_prediction, targets.confidence_target)
        if targets.confidence_target is not None
        else _zero_like(event_loss)
    )
    regime_loss = (
        compute_regime_loss(regime_prediction, targets.regime_target)
        if regime_prediction is not None and targets.regime_target is not None
        else _zero_like(event_loss)
    )

    total_loss = (
        (event_loss * loss_weights.event)
        + (boundary_loss * loss_weights.boundary)
        + (timing_loss * loss_weights.timing)
        + (confidence_loss * loss_weights.confidence)
        + (regime_loss * loss_weights.regime)
    )
    return HeadLosses(
        event_loss=event_loss,
        boundary_loss=boundary_loss,
        timing_loss=timing_loss,
        confidence_loss=confidence_loss,
        regime_loss=regime_loss,
        total_loss=total_loss,
    )


def compute_acceptance_metrics(
    *,
    event_prediction: EventPrediction,
    boundary_prediction: BoundaryPrediction,
    timing_prediction: TimingPrediction,
    confidence_prediction: ConfidencePrediction,
    targets: MultiTaskTargets,
    regime_prediction: RegimePrediction | None = None,
) -> AcceptanceMetrics:
    """Compute stable evaluation metrics for downstream offline reporting."""
    event_flag = _validate_probability_target(target=targets.event_flag, field_name="event_flag")
    future_low = _validate_float_vector(target=targets.future_low, field_name="future_low")
    future_high = _validate_float_vector(target=targets.future_high, field_name="future_high")
    event_start = _validate_float_vector(target=targets.event_start_offset, field_name="event_start_offset")
    maturity = _validate_float_vector(target=targets.maturity_offset, field_name="maturity_offset")

    event_brier = float(
        torch.mean((event_prediction.probabilities - event_flag.to(event_prediction.probabilities.device)) ** 2)
        .detach()
        .cpu()
        .item()
    )
    boundary_truth = torch.stack((future_low, future_high), dim=1).to(boundary_prediction.future_low.device)
    boundary_pred = torch.stack((boundary_prediction.future_low, boundary_prediction.future_high), dim=1)
    boundary_mae = float(torch.mean(torch.abs(boundary_pred - boundary_truth)).detach().cpu().item())

    valid_timing = (event_start >= 0.0) & (maturity >= 0.0)
    timing_mae: float | None = None
    if bool(valid_timing.any()):
        timing_truth = torch.stack((event_start[valid_timing], maturity[valid_timing]), dim=1).to(
            timing_prediction.event_start_offset.device
        )
        timing_pred = torch.stack(
            (
                timing_prediction.event_start_offset[valid_timing.to(timing_prediction.event_start_offset.device)],
                timing_prediction.maturity_offset[valid_timing.to(timing_prediction.maturity_offset.device)],
            ),
            dim=1,
        )
        timing_mae = float(torch.mean(torch.abs(timing_pred - timing_truth)).detach().cpu().item())

    confidence_brier: float | None = None
    if targets.confidence_target is not None:
        confidence_target = _validate_probability_target(
            target=targets.confidence_target,
            field_name="confidence_target",
        )
        confidence_brier = float(
            torch.mean(
                (confidence_prediction.confidence - confidence_target.to(confidence_prediction.confidence.device)) ** 2
            )
            .detach()
            .cpu()
            .item()
        )

    regime_cross_entropy: float | None = None
    if regime_prediction is not None and targets.regime_target is not None:
        regime_cross_entropy = float(
            compute_regime_loss(regime_prediction, targets.regime_target).detach().cpu().item()
        )

    return AcceptanceMetrics(
        event_brier=event_brier,
        boundary_mae=boundary_mae,
        timing_mae=timing_mae,
        confidence_brier=confidence_brier,
        regime_cross_entropy=regime_cross_entropy,
    )


def _validate_shape(prediction: torch.Tensor, target: torch.Tensor, name: str) -> None:
    if prediction.shape != target.shape:
        raise LossError(f"{name} shape mismatch: expected {tuple(target.shape)}, got {tuple(prediction.shape)}.")


def _validate_float_vector(*, target: torch.Tensor, field_name: str) -> torch.Tensor:
    if not isinstance(target, torch.Tensor):
        raise LossError(f"{field_name} must be a torch.Tensor.")
    if target.ndim != 1:
        raise LossError(f"{field_name} must have shape (batch,).")
    supported_dtypes = (
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    if target.dtype not in supported_dtypes:
        raise LossError(f"{field_name} must have a numeric dtype.")
    target_float = target.float()
    if not torch.isfinite(target_float[target_float == target_float]).all():
        raise LossError(f"{field_name} must contain only finite non-NaN values where defined.")
    return target_float


def _validate_probability_target(*, target: torch.Tensor, field_name: str) -> torch.Tensor:
    target_float = _validate_float_vector(target=target, field_name=field_name)
    if not torch.all((target_float >= 0.0) & (target_float <= 1.0)):
        raise LossError(f"{field_name} must lie in [0, 1].")
    return target_float


def _validate_class_target(*, target: torch.Tensor, field_name: str) -> torch.Tensor:
    if not isinstance(target, torch.Tensor):
        raise LossError(f"{field_name} must be a torch.Tensor.")
    if target.ndim != 1:
        raise LossError(f"{field_name} must have shape (batch,).")
    if target.dtype not in (torch.int32, torch.int64):
        raise LossError(f"{field_name} must have an integer dtype.")
    return target.long()


def _masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        device_mask = mask.to(prediction.device)
        return F.smooth_l1_loss(prediction[device_mask], target[mask].to(prediction.device))
    return _zero_like(prediction)


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _validate_loss_weights(weights: LossWeights) -> None:
    for field_name in ("event", "boundary", "timing", "confidence", "regime"):
        value = getattr(weights, field_name)
        if value < 0.0:
            raise LossError(f"{field_name} loss weight must be non-negative.")
