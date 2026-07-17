"""U10 local inference service and prediction pipeline."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from time import perf_counter

import torch

from inference.advisory import AdvisoryDecision, evaluate_advisory
from inference.model_store import LoadedActiveModel
from inference.retrieval import retrieve_analogs
from inference.runtime_features import RuntimeWindow, build_runtime_window
from models.explanation import RetrievalEvidence, render_explanation
from training.config_schema import RuntimeConfig
from training.contracts import PredictionRecordContract, PredictionRequestContract, PredictionResponseContract
from training.horizons import resolve_horizons

LOGGER = logging.getLogger(__name__)


class InferenceError(ValueError):
    """Raised when the inference pipeline fails."""


def predict(
    request: PredictionRequestContract,
    config: RuntimeConfig,
    active_model: LoadedActiveModel,
    *,
    current_time: datetime | None = None,
) -> tuple[PredictionResponseContract, ...]:
    """Run local inference for a single- or multi-horizon request.

    Args:
        request: Validated prediction request.
        config: Validated runtime configuration.
        active_model: Loaded active inference model and retrieval memory.
        current_time: Optional current time override for closed-bar validation.

    Returns:
        Tuple of typed prediction response payloads. Single-horizon requests
        return a tuple of length one.
    """
    start = perf_counter()
    runtime_window = build_runtime_window(
        request,
        config,
        active_model.lookbacks_by_timeframe,
        current_time=current_time,
    )

    model = active_model.model
    model.eval()
    with torch.no_grad():
        windows = {timeframe: tensor.to(model.device) for timeframe, tensor in runtime_window.windows_by_timeframe.items()}
        reference_close = runtime_window.reference_close.to(model.device)
        backbone_output = model.backbone(windows)
        latent = backbone_output.fused_latent

        # Determine which outputs to use based on request
        if request.requested_timeframes and request.requested_horizons:
            # Use unified outputs, extract requested combinations
            if hasattr(model, 'unified_event_head'):
                unified_event = model.unified_event_head(latent)
                unified_boundary = model.unified_boundary_head(latent)
                unified_timing = model.unified_timing_head(latent)
                # For unified heads, we still need confidence from the existing head
                confidence_prediction = model.confidence_head(latent)
                # Extract predictions for requested combinations
                predictions = _extract_unified_predictions(
                    unified_event,
                    unified_boundary,
                    unified_timing,
                    request.requested_timeframes,
                    request.requested_horizons,
                    reference_close,
                )
                # Use first prediction for evidence retrieval
                evidence = retrieve_analogs(
                    active_model.retrieval_index,
                    latent.squeeze(0).detach().cpu().numpy(),
                    query_reference_ts=runtime_window.reference_ts,
                    top_k=int(request.top_k_analogs),
                )
                advisory = evaluate_advisory(
                    confidence=float(confidence_prediction.confidence.item()),
                    evidence=evidence,
                    requested_top_k=int(request.top_k_analogs),
                )
                responses = tuple(
                    _build_prediction_response(
                        request=request,
                        runtime_window=runtime_window,
                        event_probability=float(pred['event_probability']),
                        confidence=float(confidence_prediction.confidence.item()),
                        low_price=float(pred['low_price']),
                        high_price=float(pred['high_price']),
                        start_estimate=float(pred['start_estimate']),
                        maturity_estimate=float(pred['maturity_estimate']),
                        advisory=advisory,
                        evidence=evidence,
                        horizon=pred['horizon'],
                        timeframe=pred['timeframe'],
                    )
                    for pred in predictions
                )
            else:
                # Unified heads not available, fall back to fused outputs
                raise InferenceError("Unified heads requested but not available in model")
        else:
            # Use existing fused outputs (backward compatible)
            event_prediction = model.event_head(latent)
            boundary_prediction = model.boundary_head(latent, reference_close)
            timing_prediction = model.timing_head(latent)
            confidence_prediction = model.confidence_head(latent)

            evidence = retrieve_analogs(
                active_model.retrieval_index,
                latent.squeeze(0).detach().cpu().numpy(),
                query_reference_ts=runtime_window.reference_ts,
                top_k=int(request.top_k_analogs),
            )
            advisory = evaluate_advisory(
                confidence=float(confidence_prediction.confidence.item()),
                evidence=evidence,
                requested_top_k=int(request.top_k_analogs),
            )
            horizons = resolve_horizons(config, horizon_mode=request.horizon_mode, horizon_bars=request.horizon_bars)
            responses = tuple(
                _build_prediction_response(
                    request=request,
                    runtime_window=runtime_window,
                    event_probability=float(event_prediction.probabilities.item()),
                    confidence=float(confidence_prediction.confidence.item()),
                    low_price=float(boundary_prediction.future_low.item()),
                    high_price=float(boundary_prediction.future_high.item()),
                    start_estimate=float(timing_prediction.event_start_offset.item()),
                    maturity_estimate=float(timing_prediction.maturity_offset.item()),
                    advisory=advisory,
                    evidence=evidence,
                    horizon=horizon,
                )
                for horizon in horizons
            )

    elapsed_ms = (perf_counter() - start) * 1000.0
    LOGGER.info(
        "completed_inference",
        extra={
            "event": "completed_inference",
            "reference_ts": runtime_window.reference_ts.isoformat(),
            "request_instrument_id": request.instrument_id,
            "horizon_mode": request.horizon_mode,
            "threshold": float(request.threshold),
            "latency_ms": float(round(elapsed_ms, 4)),
            "degraded_mode": bool(advisory.low_confidence_advisory),
            "model_id": active_model.checkpoint.model_id,
        },
    )
    return responses


def _build_prediction_response(
    *,
    request: PredictionRequestContract,
    runtime_window: RuntimeWindow,
    event_probability: float,
    confidence: float,
    low_price: float,
    high_price: float,
    start_estimate: float,
    maturity_estimate: float,
    advisory: AdvisoryDecision,
    evidence: RetrievalEvidence,
    horizon: int,
    timeframe: str | None = None,
) -> PredictionResponseContract:
    bounded_start = _bound_positive_offset(start_estimate, horizon)
    bounded_maturity = _bound_positive_offset(max(maturity_estimate, start_estimate), horizon)
    duration = (bounded_maturity - bounded_start) + 1

    prediction = PredictionRecordContract(
        request_id=_build_request_id(
            instrument_id=request.instrument_id,
            reference_ts=runtime_window.reference_ts,
            horizon=horizon,
            threshold=float(request.threshold),
        ),
        reference_ts=runtime_window.reference_ts,
        horizon=int(horizon),
        event_probability=float(event_probability),
        confidence=float(confidence),
        low_price=float(low_price),
        high_price=float(high_price),
        start_estimate=bounded_start,
        maturity_estimate=bounded_maturity,
        duration_estimate=duration,
        low_confidence_advisory=bool(advisory.low_confidence_advisory),
        timeframe=timeframe,
    )
    explanation = render_explanation(
        prediction,
        evidence,
        requested_top_k=int(request.top_k_analogs),
    )
    return PredictionResponseContract(
        prediction=prediction,
        top_k_analogs=explanation.top_k_analogs,
        summary_statistics=explanation.summary_statistics,
        grounded_natural_language_explanation=explanation.grounded_natural_language_explanation,
    )


def _bound_positive_offset(value: float, horizon: int) -> int:
    if horizon <= 0:
        raise InferenceError("horizon must be positive.")
    rounded = int(round(value))
    return max(1, min(int(horizon), rounded))


def _build_request_id(*, instrument_id: str, reference_ts: datetime, horizon: int, threshold: float) -> str:
    raw = f"{instrument_id}|{reference_ts.isoformat()}|{horizon}|{threshold:.8f}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"pred-{digest}"


def _extract_unified_predictions(
    unified_event: torch.Tensor,
    unified_boundary: torch.Tensor,
    unified_timing: torch.Tensor,
    requested_timeframes: tuple[str, ...],
    requested_horizons: tuple[int, ...],
    reference_close: torch.Tensor,
) -> list[dict[str, any]]:
    """Extract predictions for specific timeframe/horizon combinations from unified outputs.

    Args:
        unified_event: Event logits of shape (batch, 18).
        unified_boundary: Boundary predictions of shape (batch, 18, 2).
        unified_timing: Timing predictions of shape (batch, 18, 2).
        requested_timeframes: Tuple of requested timeframes.
        requested_horizons: Tuple of requested horizons.
        reference_close: Reference close price for boundary adjustment.

    Returns:
        List of prediction dictionaries for each requested combination.
    """
    # Output order: 1m_h15, 1m_h60, 1m_h120, 5m_h15, 5m_h60, 5m_h120, ..., 1d_h120
    timeframes = ("1m", "5m", "15m", "1h", "4h", "1d")
    horizons = (15, 60, 120)

    predictions = []
    ref_close = reference_close.item()

    for timeframe in requested_timeframes:
        if timeframe not in timeframes:
            continue

        tf_idx = timeframes.index(timeframe)
        for horizon in requested_horizons:
            if horizon not in horizons:
                continue

            h_idx = horizons.index(horizon)
            flat_idx = tf_idx * 3 + h_idx

            # Extract event probability
            event_logit = unified_event[0, flat_idx]
            event_probability = float(torch.sigmoid(event_logit).item())

            # Extract boundary predictions
            boundary_pred = unified_boundary[0, flat_idx, :]
            low_price = float(boundary_pred[0].item())
            high_price = float(boundary_pred[1].item())

            # Adjust boundaries relative to reference close
            low_price = ref_close - abs(low_price)
            high_price = ref_close + abs(high_price)

            # Extract timing predictions
            timing_pred = unified_timing[0, flat_idx, :]
            start_estimate = float(timing_pred[0].item())
            maturity_estimate = float(timing_pred[1].item())

            predictions.append({
                'event_probability': event_probability,
                'low_price': low_price,
                'high_price': high_price,
                'start_estimate': start_estimate,
                'maturity_estimate': maturity_estimate,
                'horizon': horizon,
                'timeframe': timeframe,
            })

    return predictions
