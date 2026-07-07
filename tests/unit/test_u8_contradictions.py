"""Unit tests for U8 contradiction prevention."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from models.explanation import ExplanationError, RetrievalAudit, RetrievalEvidence, render_explanation
from training.contracts import AnalogRecordContract, PredictionRecordContract


def test_u8_low_confidence_advisory_uses_limited_support_wording() -> None:
    evidence = RetrievalEvidence(
        analogs=(
            AnalogRecordContract(
                analog_id="fold-03:2026-01-01T00:00:00+00:00",
                reference_ts=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
                distance=0.25,
                outcome_summary="event_observed=1; future_low=98.0000; future_high=102.0000; duration_bars=2.0",
            ),
        ),
        summary_statistics={
            "analog_count": 1.0,
            "mean_distance": 0.25,
            "observed_event_rate": 1.0,
            "mean_future_low": 98.0,
            "mean_future_high": 102.0,
            "mean_boundary_span": 4.0,
            "mean_duration_bars": 2.0,
        },
        audit=RetrievalAudit(
            index_scope="train_only",
            query_reference_ts=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
            candidate_count=1,
            filtered_future_count=0,
            returned_count=1,
        ),
    )
    prediction = PredictionRecordContract(
        request_id="req-2",
        reference_ts=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        horizon=5,
        event_probability=0.55,
        confidence=0.40,
        low_price=98.1,
        high_price=102.1,
        start_estimate=2,
        maturity_estimate=3,
        duration_estimate=2,
        low_confidence_advisory=True,
    )

    output = render_explanation(prediction, evidence, requested_top_k=2)

    assert "limited" in output.grounded_natural_language_explanation.lower()
    assert "supports the current event probability" not in output.grounded_natural_language_explanation.lower()


def test_u8_high_confidence_prediction_cannot_be_unexplained() -> None:
    evidence = RetrievalEvidence(
        analogs=tuple(),
        summary_statistics={"analog_count": 0.0, "mean_distance": 0.0},
        audit=RetrievalAudit(
            index_scope="train_only",
            query_reference_ts=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
            candidate_count=0,
            filtered_future_count=0,
            returned_count=0,
        ),
    )
    prediction = PredictionRecordContract(
        request_id="req-3",
        reference_ts=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        horizon=5,
        event_probability=0.90,
        confidence=0.91,
        low_price=97.5,
        high_price=103.5,
        start_estimate=1,
        maturity_estimate=3,
        duration_estimate=3,
        low_confidence_advisory=False,
    )

    with pytest.raises(ExplanationError, match="at least one analog is required"):
        render_explanation(prediction, evidence, requested_top_k=2)
