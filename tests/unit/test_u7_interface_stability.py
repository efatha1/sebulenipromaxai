"""Unit tests for U7 typed interface stability."""

from __future__ import annotations

from dataclasses import fields

import torch

from models.boundary_head import BoundaryHead, BoundaryPrediction, predict_boundaries
from models.confidence_head import ConfidenceHead, ConfidencePrediction, predict_confidence
from models.event_head import EventHead, EventPrediction, predict_event
from models.losses import AcceptanceMetrics, HeadLosses
from models.regime_head import RegimeHead, RegimePrediction, score_regime
from models.timing_head import TimingHead, TimingPrediction, predict_timing


def test_u7_prediction_dataclasses_expose_stable_fields() -> None:
    assert [field.name for field in fields(EventPrediction)] == ["logits", "probabilities"]
    assert [field.name for field in fields(BoundaryPrediction)] == [
        "lower_delta",
        "upper_delta",
        "future_low",
        "future_high",
    ]
    assert [field.name for field in fields(TimingPrediction)] == ["event_start_offset", "maturity_offset"]
    assert [field.name for field in fields(ConfidencePrediction)] == ["logits", "confidence"]
    assert [field.name for field in fields(RegimePrediction)] == [
        "logits",
        "probabilities",
        "predicted_regime",
    ]
    assert [field.name for field in fields(HeadLosses)] == [
        "event_loss",
        "boundary_loss",
        "timing_loss",
        "confidence_loss",
        "regime_loss",
        "total_loss",
    ]
    assert [field.name for field in fields(AcceptanceMetrics)] == [
        "event_brier",
        "boundary_mae",
        "timing_mae",
        "confidence_brier",
        "regime_cross_entropy",
    ]


def test_u7_functional_interfaces_return_typed_outputs() -> None:
    latent = torch.randn(3, 7)
    reference_close = torch.full((3,), 100.0)

    event_output = predict_event(EventHead(latent_dim=7), latent)
    boundary_output = predict_boundaries(BoundaryHead(latent_dim=7), latent, reference_close)
    timing_output = predict_timing(TimingHead(latent_dim=7, max_horizon_bars=8), latent)
    confidence_output = predict_confidence(ConfidenceHead(latent_dim=7), latent)
    regime_output = score_regime(RegimeHead(latent_dim=7, num_regimes=2), latent)

    assert isinstance(event_output, EventPrediction)
    assert isinstance(boundary_output, BoundaryPrediction)
    assert isinstance(timing_output, TimingPrediction)
    assert isinstance(confidence_output, ConfidencePrediction)
    assert isinstance(regime_output, RegimePrediction)
