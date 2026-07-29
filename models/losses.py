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
class MultiTaskTargetsUnified:
    """Unified targets for all (timeframe, horizon, threshold) combinations.
    
    Tensor structure:
    - Each field is a dict keyed by timeframe (e.g., "1m", "5m", "1h", "4h", "1d")
    - Each tensor has shape (batch, num_horizons) for the current single-threshold configuration
    - Future multi-threshold support would add a third dimension: (batch, num_horizons, num_thresholds)
    
    Example with horizons [15, 60, 120]:
    - targets.event_flag["1m"] has shape (batch, 3)
    - targets.event_flag["1m"][:, 0] corresponds to horizon 15
    - targets.event_flag["1m"][:, 1] corresponds to horizon 60
    - targets.event_flag["1m"][:, 2] corresponds to horizon 120
    """

    event_flag: dict[str, torch.Tensor]
    future_low: dict[str, torch.Tensor]
    future_high: dict[str, torch.Tensor]
    event_start_offset: dict[str, torch.Tensor]
    maturity_offset: dict[str, torch.Tensor]


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
class UncertaintyWeightedLosses:
    """Typed per-task and aggregate loss outputs with uncertainty parameters."""

    event_loss: torch.Tensor
    boundary_loss: torch.Tensor
    timing_loss: torch.Tensor
    total_loss: torch.Tensor
    event_log_sigma: torch.Tensor  # Shape: (18,)
    boundary_log_sigma: torch.Tensor  # Shape: (18,)
    timing_log_sigma: torch.Tensor  # Shape: (18,)


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


def compute_unified_multitask_loss(
    *,
    unified_event: torch.Tensor,
    unified_boundary: torch.Tensor,
    unified_timing: torch.Tensor,
    targets: MultiTaskTargetsUnified,
    requested_outputs: list[tuple[str, int]] | None = None,
) -> HeadLosses:
    """Compute multi-task loss for unified 18-output heads.

    Args:
        unified_event: Event logits of shape (batch, 18).
        unified_boundary: Boundary predictions of shape (batch, 18, 2).
        unified_timing: Timing predictions of shape (batch, 18, 2).
        targets: Unified targets for all 18 combinations.
        requested_outputs: Optional list of (timeframe, horizon) to compute loss for.
                          If None, computes loss for all 18 combinations.

    Returns:
        HeadLosses with aggregated losses across all combinations.
    """
    # Reshape unified outputs to separate by timeframe
    # Output order: 1m_h15, 1m_h60, 1m_h120, 5m_h15, 5m_h60, 5m_h120, ..., 1d_h120
    timeframes = ("1m", "5m", "15m", "1h", "4h", "1d")
    horizons = (15, 60, 120)

    event_losses = []
    boundary_losses = []
    timing_losses = []

    for tf_idx, timeframe in enumerate(timeframes):
        if requested_outputs is not None:
            # Only compute loss for requested combinations
            requested_horizons_for_tf = [h for t, h in requested_outputs if t == timeframe]
            if not requested_horizons_for_tf:
                continue
        else:
            requested_horizons_for_tf = horizons

        for h_idx, horizon in enumerate(horizons):
            if requested_outputs is not None and horizon not in requested_horizons_for_tf:
                continue

            # Calculate flat index in the 18-output tensor
            flat_idx = tf_idx * 3 + h_idx

            # Extract event logits for this combination
            event_logits = unified_event[:, flat_idx : flat_idx + 1]
            event_target = targets.event_flag[timeframe][:, h_idx : h_idx + 1]
            event_loss = F.binary_cross_entropy_with_logits(event_logits, event_target)
            event_losses.append(event_loss)

            # Extract boundary predictions for this combination
            boundary_pred = unified_boundary[:, flat_idx, :]
            boundary_low = targets.future_low[timeframe][:, h_idx : h_idx + 1]
            boundary_high = targets.future_high[timeframe][:, h_idx : h_idx + 1]
            boundary_truth = torch.stack((boundary_low, boundary_high), dim=1).to(device=boundary_pred.device, dtype=boundary_pred.dtype)
            boundary_loss = F.smooth_l1_loss(boundary_pred, boundary_truth)
            boundary_losses.append(boundary_loss)

            # Extract timing predictions for this combination
            timing_pred = unified_timing[:, flat_idx, :]
            timing_start = targets.event_start_offset[timeframe][:, h_idx : h_idx + 1]
            timing_maturity = targets.maturity_offset[timeframe][:, h_idx : h_idx + 1]
            timing_truth = torch.stack((timing_start, timing_maturity), dim=1).to(device=timing_pred.device, dtype=timing_pred.dtype)

            # Mask invalid timing targets
            valid_start = timing_start >= 0.0
            valid_maturity = timing_maturity >= 0.0
            valid_mask = (valid_start & valid_maturity).to(timing_pred.device)

            if valid_mask.any():
                timing_loss = F.smooth_l1_loss(timing_pred[valid_mask], timing_truth[valid_mask])
            else:
                timing_loss = torch.tensor(0.0, device=timing_pred.device)
            timing_losses.append(timing_loss)

    # Aggregate losses
    total_event_loss = torch.stack(event_losses).mean() if event_losses else torch.tensor(0.0, device=unified_event.device)
    total_boundary_loss = torch.stack(boundary_losses).mean() if boundary_losses else torch.tensor(0.0, device=unified_boundary.device)
    total_timing_loss = torch.stack(timing_losses).mean() if timing_losses else torch.tensor(0.0, device=unified_timing.device)

    total_loss = total_event_loss + total_boundary_loss + total_timing_loss

    return HeadLosses(
        event_loss=total_event_loss,
        boundary_loss=total_boundary_loss,
        timing_loss=total_timing_loss,
        confidence_loss=torch.tensor(0.0, device=unified_event.device),
        regime_loss=torch.tensor(0.0, device=unified_event.device),
        total_loss=total_loss,
    )


def compute_unified_multitask_loss_with_uncertainty(
    *,
    unified_event: torch.Tensor,
    unified_boundary: torch.Tensor,
    unified_timing: torch.Tensor,
    event_log_sigma: torch.Tensor,
    boundary_log_sigma: torch.Tensor,
    timing_log_sigma: torch.Tensor,
    targets: MultiTaskTargetsUnified,
    requested_outputs: list[tuple[str, int]] | None = None,
) -> UncertaintyWeightedLosses:
    """Compute multi-task loss for unified 18-output heads with homoscedastic uncertainty weighting.

    Implements the loss function: L_total = Σ(1/(2σ_i²) * L_i + log(σ_i))
    where σ_i is a learnable uncertainty parameter for each of the 18 targets.

    Args:
        unified_event: Event logits of shape (batch, 18).
        unified_boundary: Boundary predictions of shape (batch, 18, 2).
        unified_timing: Timing predictions of shape (batch, 18, 2).
        event_log_sigma: Log uncertainty parameters for event head, shape (18,).
        boundary_log_sigma: Log uncertainty parameters for boundary head, shape (18,).
        timing_log_sigma: Log uncertainty parameters for timing head, shape (18,).
        targets: Unified targets for all 18 combinations.
        requested_outputs: Optional list of (timeframe, horizon) to compute loss for.
                          If None, computes loss for all 18 combinations.

    Returns:
        UncertaintyWeightedLosses with aggregated losses and uncertainty parameters.
    """
    # Reshape unified outputs to separate by timeframe
    # Output order: 1m_h15, 1m_h60, 1m_h120, 5m_h15, 5m_h60, 5m_h120, ..., 1d_h120
    timeframes = ("1m", "5m", "15m", "1h", "4h", "1d")
    horizons = (15, 60, 120)

    event_losses = []
    boundary_losses = []
    timing_losses = []

    for tf_idx, timeframe in enumerate(timeframes):
        if requested_outputs is not None:
            # Only compute loss for requested combinations
            requested_horizons_for_tf = [h for t, h in requested_outputs if t == timeframe]
            if not requested_horizons_for_tf:
                continue
        else:
            requested_horizons_for_tf = horizons

        for h_idx, horizon in enumerate(horizons):
            if requested_outputs is not None and horizon not in requested_horizons_for_tf:
                continue

            # Calculate flat index in the 18-output tensor
            flat_idx = tf_idx * 3 + h_idx

            # Extract event logits for this combination
            event_logits = unified_event[:, flat_idx : flat_idx + 1]
            event_target = targets.event_flag[timeframe][:, h_idx : h_idx + 1]
            event_loss = F.binary_cross_entropy_with_logits(event_logits, event_target)
            event_losses.append(event_loss)

            # Extract boundary predictions for this combination
            boundary_pred = unified_boundary[:, flat_idx, :]
            boundary_low = targets.future_low[timeframe][:, h_idx : h_idx + 1]
            boundary_high = targets.future_high[timeframe][:, h_idx : h_idx + 1]
            boundary_truth = torch.stack((boundary_low, boundary_high), dim=1).to(device=boundary_pred.device, dtype=boundary_pred.dtype)
            boundary_loss = F.smooth_l1_loss(boundary_pred, boundary_truth)
            boundary_losses.append(boundary_loss)

            # Extract timing predictions for this combination
            timing_pred = unified_timing[:, flat_idx, :]
            timing_start = targets.event_start_offset[timeframe][:, h_idx : h_idx + 1]
            timing_maturity = targets.maturity_offset[timeframe][:, h_idx : h_idx + 1]
            timing_truth = torch.stack((timing_start, timing_maturity), dim=1).to(device=timing_pred.device, dtype=timing_pred.dtype)

            # Mask invalid timing targets
            valid_start = timing_start >= 0.0
            valid_maturity = timing_maturity >= 0.0
            valid_mask = (valid_start & valid_maturity).to(timing_pred.device)

            if valid_mask.any():
                timing_loss = F.smooth_l1_loss(timing_pred[valid_mask], timing_truth[valid_mask])
            else:
                timing_loss = torch.tensor(0.0, device=timing_pred.device)
            timing_losses.append(timing_loss)

    # Stack losses per combination (18 losses per task)
    event_losses_stacked = torch.stack(event_losses)  # Shape: (18,)
    boundary_losses_stacked = torch.stack(boundary_losses)  # Shape: (18,)
    timing_losses_stacked = torch.stack(timing_losses)  # Shape: (18,)

    # Compute uncertainty-weighted loss: L_total = Σ(1/(2σ²) * L + log(σ))
    # For event head
    event_weights = torch.exp(-2 * event_log_sigma)  # 1/σ²
    event_weighted_loss = (0.5 * event_weights * event_losses_stacked + event_log_sigma).sum()

    # For boundary head
    boundary_weights = torch.exp(-2 * boundary_log_sigma)  # 1/σ²
    boundary_weighted_loss = (0.5 * boundary_weights * boundary_losses_stacked + boundary_log_sigma).sum()

    # For timing head
    timing_weights = torch.exp(-2 * timing_log_sigma)  # 1/σ²
    timing_weighted_loss = (0.5 * timing_weights * timing_losses_stacked + timing_log_sigma).sum()

    total_loss = event_weighted_loss + boundary_weighted_loss + timing_weighted_loss

    return UncertaintyWeightedLosses(
        event_loss=event_weighted_loss,
        boundary_loss=boundary_weighted_loss,
        timing_loss=timing_weighted_loss,
        total_loss=total_loss,
        event_log_sigma=event_log_sigma,
        boundary_log_sigma=boundary_log_sigma,
        timing_log_sigma=timing_log_sigma,
    )
