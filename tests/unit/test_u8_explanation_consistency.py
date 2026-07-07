"""Unit tests for deterministic grounded explanation behavior."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from inference.retrieval import build_retrieval_index, retrieve_analogs
from models.explanation import render_explanation
from training.contracts import PredictionRecordContract
from training.latent_export import export_training_latents


def test_u8_explanation_is_consistent_for_identical_input() -> None:
    memory_rows = export_training_latents(
        latent_matrix=np.asarray([[0.0, 0.0], [0.5, 1.0], [1.0, 1.5]], dtype=np.float64),
        reference_ts=[
            datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
        ],
        event_observed=[0.0, 1.0, 1.0],
        future_low=[99.0, 98.5, 98.0],
        future_high=[101.0, 102.5, 103.0],
        event_start_offset=[None, 2, 2],
        maturity_offset=[None, 3, 4],
        source_fold_id="fold-02",
        source_split="train",
    )
    index = build_retrieval_index(memory_rows)
    evidence = retrieve_analogs(
        index,
        np.asarray([0.75, 1.25], dtype=np.float64),
        query_reference_ts=datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
        top_k=2,
    )
    prediction = PredictionRecordContract(
        request_id="req-1",
        reference_ts=datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
        horizon=5,
        event_probability=0.82,
        confidence=0.84,
        low_price=98.2,
        high_price=103.1,
        start_estimate=2,
        maturity_estimate=4,
        duration_estimate=3,
        low_confidence_advisory=False,
    )

    first = render_explanation(prediction, evidence, requested_top_k=2)
    second = render_explanation(prediction, evidence, requested_top_k=2)

    assert first.grounded_natural_language_explanation == second.grounded_natural_language_explanation
    assert first.summary_statistics == second.summary_statistics
    assert [analog.analog_id for analog in first.top_k_analogs] == [analog.analog_id for analog in second.top_k_analogs]
    assert "top 2 training analogs" in first.grounded_natural_language_explanation.lower()
    assert "0.82" in first.grounded_natural_language_explanation
