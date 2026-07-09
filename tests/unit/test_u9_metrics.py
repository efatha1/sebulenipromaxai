"""Unit tests for U9 metric helpers."""

from __future__ import annotations

import torch

from models.losses import AcceptanceMetrics, HeadLosses
from training.metrics import aggregate_metric_dicts, summarize_split_metrics


def test_u9_metrics_summary_and_aggregation() -> None:
    losses = HeadLosses(
        event_loss=torch.tensor(0.1),
        boundary_loss=torch.tensor(0.2),
        timing_loss=torch.tensor(0.3),
        confidence_loss=torch.tensor(0.4),
        regime_loss=torch.tensor(0.0),
        total_loss=torch.tensor(1.0),
    )
    acceptance = AcceptanceMetrics(
        event_brier=0.05,
        boundary_mae=0.15,
        timing_mae=1.25,
        confidence_brier=0.08,
        regime_cross_entropy=None,
    )

    first = summarize_split_metrics(
        sample_count=8,
        losses=losses,
        acceptance_metrics=acceptance,
        explanation_coverage_rate=1.0,
    )
    second = dict(first)
    second["total_loss"] = 3.0

    aggregate = aggregate_metric_dicts([first, second])

    assert first["sample_count"] == 8.0
    assert first["event_loss"] == 0.1
    assert first["timing_mae"] == 1.25
    assert first["confidence_brier"] == 0.08
    assert first["regime_cross_entropy"] == -1.0
    assert first["explanation_coverage_rate"] == 1.0
    assert aggregate["total_loss"] == 2.0
