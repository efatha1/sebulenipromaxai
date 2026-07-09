"""Unit tests for U7 head output shapes."""

from __future__ import annotations

import torch

from models.boundary_head import BoundaryHead
from models.confidence_head import ConfidenceHead
from models.event_head import EventHead
from models.regime_head import RegimeHead
from models.timing_head import TimingHead


def test_u7_head_output_shapes() -> None:
    latent_dim = 8
    batch_size = 5
    max_horizon_bars = 12
    latent = torch.randn(batch_size, latent_dim)
    reference_close = torch.full((batch_size,), 100.0)

    event_head = EventHead(latent_dim=latent_dim)
    boundary_head = BoundaryHead(latent_dim=latent_dim)
    timing_head = TimingHead(latent_dim=latent_dim, max_horizon_bars=max_horizon_bars)
    confidence_head = ConfidenceHead(latent_dim=latent_dim)
    regime_head = RegimeHead(latent_dim=latent_dim, num_regimes=3)

    event_output = event_head(latent)
    boundary_output = boundary_head(latent, reference_close)
    timing_output = timing_head(latent)
    confidence_output = confidence_head(latent)
    regime_output = regime_head(latent)

    assert event_output.logits.shape == (batch_size,)
    assert event_output.probabilities.shape == (batch_size,)

    assert boundary_output.lower_delta.shape == (batch_size,)
    assert boundary_output.upper_delta.shape == (batch_size,)
    assert boundary_output.future_low.shape == (batch_size,)
    assert boundary_output.future_high.shape == (batch_size,)

    assert timing_output.event_start_offset.shape == (batch_size,)
    assert timing_output.maturity_offset.shape == (batch_size,)

    assert confidence_output.logits.shape == (batch_size,)
    assert confidence_output.confidence.shape == (batch_size,)

    assert regime_output.logits.shape == (batch_size, 3)
    assert regime_output.probabilities.shape == (batch_size, 3)
    assert regime_output.predicted_regime.shape == (batch_size,)
