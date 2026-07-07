"""Unit tests for U7 output ranges and bounded behavior."""

from __future__ import annotations

import torch

from models.boundary_head import BoundaryHead
from models.confidence_head import ConfidenceHead, confidence_from_logit
from models.event_head import EventHead
from models.regime_head import RegimeHead
from models.timing_head import TimingHead


def test_u7_outputs_stay_in_expected_ranges() -> None:
    latent_dim = 6
    batch_size = 4
    max_horizon_bars = 9
    latent = torch.randn(batch_size, latent_dim)
    reference_close = torch.full((batch_size,), 100.0)

    event_output = EventHead(latent_dim=latent_dim)(latent)
    boundary_output = BoundaryHead(latent_dim=latent_dim)(latent, reference_close)
    timing_output = TimingHead(latent_dim=latent_dim, max_horizon_bars=max_horizon_bars)(latent)
    confidence_output = ConfidenceHead(latent_dim=latent_dim)(latent)
    regime_output = RegimeHead(latent_dim=latent_dim, num_regimes=4)(latent)

    assert torch.all((event_output.probabilities >= 0.0) & (event_output.probabilities <= 1.0))
    assert torch.all(boundary_output.future_low <= boundary_output.future_high)
    assert torch.all(timing_output.event_start_offset >= 1.0)
    assert torch.all(timing_output.event_start_offset <= float(max_horizon_bars))
    assert torch.all(timing_output.maturity_offset >= timing_output.event_start_offset)
    assert torch.all(timing_output.maturity_offset <= float(max_horizon_bars))
    assert torch.all((confidence_output.confidence >= 0.0) & (confidence_output.confidence <= 1.0))
    assert torch.all((regime_output.probabilities >= 0.0) & (regime_output.probabilities <= 1.0))
    assert torch.allclose(
        regime_output.probabilities.sum(dim=1),
        torch.ones(batch_size),
        atol=1e-6,
    )


def test_confidence_from_logit_is_monotonic() -> None:
    low_logits = torch.tensor([-2.0, -0.5, 0.25])
    high_logits = torch.tensor([-1.5, 0.0, 1.5])

    low_confidence = confidence_from_logit(low_logits)
    high_confidence = confidence_from_logit(high_logits)

    assert torch.all(high_confidence > low_confidence)
