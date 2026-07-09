"""Unit tests for deterministic U8 retrieval behavior."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from inference.retrieval import build_retrieval_index, retrieve_analogs
from training.latent_export import export_training_latents


def test_u8_retrieval_is_deterministic_and_stably_tie_broken() -> None:
    reference_ts = [
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
    ]
    latent_matrix = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [5.0, 5.0],
        ],
        dtype=np.float64,
    )
    memory_rows = export_training_latents(
        latent_matrix=latent_matrix,
        reference_ts=reference_ts,
        event_observed=[0.0, 1.0, 1.0],
        future_low=[99.0, 98.0, 97.5],
        future_high=[101.0, 103.0, 104.0],
        event_start_offset=[None, 2, 1],
        maturity_offset=[None, 3, 2],
        source_fold_id="fold-01",
        source_split="train",
    )
    index = build_retrieval_index(memory_rows)
    query_reference_ts = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    query_latent = np.asarray([0.0, 1.0], dtype=np.float64)

    first = retrieve_analogs(index, query_latent, query_reference_ts=query_reference_ts, top_k=2)
    second = retrieve_analogs(index, query_latent, query_reference_ts=query_reference_ts, top_k=2)

    assert [analog.analog_id for analog in first.analogs] == [analog.analog_id for analog in second.analogs]
    assert [analog.analog_id for analog in first.analogs] == [
        "fold-01:2026-01-01T00:00:00+00:00",
        "fold-01:2026-01-01T00:01:00+00:00",
    ]
    assert [analog.distance for analog in first.analogs] == [analog.distance for analog in second.analogs]
    assert first.summary_statistics == second.summary_statistics
    assert first.audit.returned_count == 2
    assert first.audit.filtered_future_count == 0
