"""Unit tests for U7 loss behavior."""

from __future__ import annotations

import torch

from models.boundary_head import BoundaryPrediction
from models.confidence_head import ConfidencePrediction
from models.event_head import EventPrediction
from models.losses import MultiTaskTargets, compute_multitask_loss
from models.regime_head import RegimePrediction
from models.timing_head import TimingPrediction


def test_multitask_loss_rewards_better_predictions() -> None:
    targets = MultiTaskTargets(
        event_flag=torch.tensor([0.0, 1.0]),
        future_low=torch.tensor([98.0, 99.5]),
        future_high=torch.tensor([101.0, 103.0]),
        event_start_offset=torch.tensor([1.0, 2.0]),
        maturity_offset=torch.tensor([2.0, 4.0]),
        confidence_target=torch.tensor([0.1, 0.9]),
        regime_target=torch.tensor([0, 1], dtype=torch.int64),
    )

    good_event = EventPrediction(
        logits=torch.tensor([-6.0, 6.0]),
        probabilities=torch.sigmoid(torch.tensor([-6.0, 6.0])),
    )
    bad_event = EventPrediction(
        logits=torch.tensor([6.0, -6.0]),
        probabilities=torch.sigmoid(torch.tensor([6.0, -6.0])),
    )

    good_boundary = BoundaryPrediction(
        lower_delta=torch.tensor([-2.0, -0.5]),
        upper_delta=torch.tensor([1.0, 3.0]),
        future_low=torch.tensor([98.0, 99.5]),
        future_high=torch.tensor([101.0, 103.0]),
    )
    bad_boundary = BoundaryPrediction(
        lower_delta=torch.tensor([3.0, 2.0]),
        upper_delta=torch.tensor([5.0, 6.0]),
        future_low=torch.tensor([103.0, 102.0]),
        future_high=torch.tensor([105.0, 107.0]),
    )

    good_timing = TimingPrediction(
        event_start_offset=torch.tensor([1.0, 2.0]),
        maturity_offset=torch.tensor([2.0, 4.0]),
    )
    bad_timing = TimingPrediction(
        event_start_offset=torch.tensor([4.0, 4.0]),
        maturity_offset=torch.tensor([5.0, 5.0]),
    )

    good_confidence = ConfidencePrediction(
        logits=torch.tensor([-2.2, 2.2]),
        confidence=torch.tensor([0.1, 0.9]),
    )
    bad_confidence = ConfidencePrediction(
        logits=torch.tensor([2.2, -2.2]),
        confidence=torch.tensor([0.9, 0.1]),
    )

    good_regime = RegimePrediction(
        logits=torch.tensor([[5.0, -5.0], [-5.0, 5.0]]),
        probabilities=torch.tensor([[0.9999, 0.0001], [0.0001, 0.9999]]),
        predicted_regime=torch.tensor([0, 1]),
    )
    bad_regime = RegimePrediction(
        logits=torch.tensor([[-5.0, 5.0], [5.0, -5.0]]),
        probabilities=torch.tensor([[0.0001, 0.9999], [0.9999, 0.0001]]),
        predicted_regime=torch.tensor([1, 0]),
    )

    good_losses = compute_multitask_loss(
        event_prediction=good_event,
        boundary_prediction=good_boundary,
        timing_prediction=good_timing,
        confidence_prediction=good_confidence,
        regime_prediction=good_regime,
        targets=targets,
    )
    bad_losses = compute_multitask_loss(
        event_prediction=bad_event,
        boundary_prediction=bad_boundary,
        timing_prediction=bad_timing,
        confidence_prediction=bad_confidence,
        regime_prediction=bad_regime,
        targets=targets,
    )

    assert good_losses.event_loss < bad_losses.event_loss
    assert good_losses.boundary_loss < bad_losses.boundary_loss
    assert good_losses.timing_loss < bad_losses.timing_loss
    assert good_losses.confidence_loss < bad_losses.confidence_loss
    assert good_losses.regime_loss < bad_losses.regime_loss
    assert good_losses.total_loss < bad_losses.total_loss
