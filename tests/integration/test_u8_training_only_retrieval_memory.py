"""Integration tests for U8 training-only retrieval memory."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from inference.retrieval import build_retrieval_index, retrieve_analogs
from models.backbone import Backbone
from models.boundary_head import BoundaryHead
from models.common import TIMEFRAMES
from models.confidence_head import ConfidenceHead
from models.event_head import EventHead
from models.explanation import render_explanation
from models.timing_head import TimingHead
from training.contracts import PredictionRecordContract
from training.latent_export import export_training_latents
from training.windowing import build_windows


def test_u8_uses_training_only_memory_for_evaluation_explanations() -> None:
    torch.manual_seed(13)
    feature_names = ("f1", "f2", "f3", "f4")
    tz = timezone.utc

    features_by_tf: dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        rows = 12
        end_ts = pd.DatetimeIndex([datetime(2026, 1, 1, 0, i, tzinfo=tz) for i in range(rows)])
        data = np.random.RandomState(2).randn(rows, len(feature_names)).astype("float64")
        frame = pd.DataFrame(data, columns=feature_names, index=end_ts)
        frame.index.name = "end_ts"
        features_by_tf[tf] = frame

    windows = build_windows(features_by_tf, lookbacks_by_timeframe={tf: 3 for tf in TIMEFRAMES})
    windows_by_tf = {tf: torch.tensor(windows[tf].windows, dtype=torch.float32) for tf in TIMEFRAMES}
    feature_dim = int(windows_by_tf["1m"].shape[2])

    backbone = Backbone(
        feature_dim=feature_dim,
        seed=21,
        device_preference="cpu",
        allow_nondeterministic=False,
    )
    fused_latent = backbone(windows_by_tf).fused_latent.detach().cpu()
    reference_ts = list(windows["1m"].reference_ts.to_pydatetime())

    train_count = 6
    memory_rows = export_training_latents(
        latent_matrix=fused_latent[:train_count].numpy(),
        reference_ts=reference_ts[:train_count],
        event_observed=[float(index % 2) for index in range(train_count)],
        future_low=[98.0 + (index * 0.1) for index in range(train_count)],
        future_high=[101.0 + (index * 0.2) for index in range(train_count)],
        event_start_offset=[None if index % 2 == 0 else 2 for index in range(train_count)],
        maturity_offset=[None if index % 2 == 0 else 4 for index in range(train_count)],
        source_fold_id="fold-04",
        source_split="train",
    )
    index = build_retrieval_index(memory_rows)

    query_index = 8
    query_ts = reference_ts[query_index]
    query_latent = fused_latent[query_index : query_index + 1]

    event_output = EventHead(latent_dim=feature_dim)(query_latent)
    boundary_output = BoundaryHead(latent_dim=feature_dim)(
        query_latent,
        torch.tensor([100.0], dtype=torch.float32),
    )
    timing_output = TimingHead(latent_dim=feature_dim, max_horizon_bars=5)(query_latent)
    confidence_output = ConfidenceHead(latent_dim=feature_dim)(query_latent)

    start_estimate = int(round(float(timing_output.event_start_offset.item())))
    maturity_estimate = int(round(float(timing_output.maturity_offset.item())))
    prediction = PredictionRecordContract(
        request_id="req-u8-int",
        reference_ts=query_ts,
        horizon=5,
        event_probability=float(event_output.probabilities.item()),
        confidence=float(confidence_output.confidence.item()),
        low_price=float(boundary_output.future_low.item()),
        high_price=float(boundary_output.future_high.item()),
        start_estimate=start_estimate,
        maturity_estimate=maturity_estimate,
        duration_estimate=(maturity_estimate - start_estimate) + 1,
        low_confidence_advisory=bool(confidence_output.confidence.item() < 0.5),
    )

    evidence = retrieve_analogs(
        index,
        query_latent.squeeze(0).numpy(),
        query_reference_ts=query_ts,
        top_k=3,
    )
    explanation = render_explanation(prediction, evidence, requested_top_k=3)

    assert explanation.audit.index_scope == "train_only"
    assert explanation.audit.returned_count == 3
    assert all(analog.reference_ts < query_ts for analog in explanation.top_k_analogs)
    assert explanation.summary_statistics["analog_count"] == 3.0
    assert explanation.grounded_natural_language_explanation
