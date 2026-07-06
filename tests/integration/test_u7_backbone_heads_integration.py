"""Integration tests for U7 heads on top of the U6 backbone."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from models.backbone import Backbone
from models.boundary_head import BoundaryHead
from models.common import TIMEFRAMES
from models.confidence_head import ConfidenceHead
from models.event_head import EventHead
from models.losses import MultiTaskTargets, compute_acceptance_metrics, compute_multitask_loss
from models.regime_head import RegimeHead
from models.timing_head import TimingHead
from training.windowing import build_windows


def test_u7_backbone_plus_heads_backward_pass() -> None:
    torch.manual_seed(7)
    feature_names = ("f1", "f2", "f3", "f4")
    tz = timezone.utc

    features_by_tf: dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        rows = 12
        end_ts = pd.DatetimeIndex([datetime(2026, 1, 1, 0, i, tzinfo=tz) for i in range(rows)])
        data = np.random.RandomState(1).randn(rows, len(feature_names)).astype("float64")
        frame = pd.DataFrame(data, columns=feature_names, index=end_ts)
        frame.index.name = "end_ts"
        features_by_tf[tf] = frame

    windows = build_windows(features_by_tf, lookbacks_by_timeframe={tf: 3 for tf in TIMEFRAMES})
    windows_by_tf = {tf: torch.tensor(windows[tf].windows, dtype=torch.float32) for tf in TIMEFRAMES}

    batch_size, _, feature_dim = windows_by_tf["1m"].shape
    backbone = Backbone(
        feature_dim=int(feature_dim),
        seed=17,
        device_preference="cpu",
        allow_nondeterministic=False,
    )
    event_head = EventHead(latent_dim=int(feature_dim))
    boundary_head = BoundaryHead(latent_dim=int(feature_dim))
    timing_head = TimingHead(latent_dim=int(feature_dim), max_horizon_bars=5)
    confidence_head = ConfidenceHead(latent_dim=int(feature_dim))
    regime_head = RegimeHead(latent_dim=int(feature_dim), num_regimes=2)

    backbone_output = backbone(windows_by_tf)
    reference_close = torch.full((batch_size,), 100.0)

    event_prediction = event_head(backbone_output.fused_latent)
    boundary_prediction = boundary_head(backbone_output.fused_latent, reference_close)
    timing_prediction = timing_head(backbone_output.fused_latent)
    confidence_prediction = confidence_head(backbone_output.fused_latent)
    regime_prediction = regime_head(backbone_output.fused_latent)

    alternating_event = torch.tensor([(index % 2) for index in range(batch_size)], dtype=torch.float32)
    start_offsets = torch.tensor(
        [float((index % 5) + 1) if bool(alternating_event[index].item()) else -1.0 for index in range(batch_size)]
    )
    maturity_offsets = torch.tensor(
        [
            min(5.0, start_offsets[index].item() + 1.0) if start_offsets[index].item() > 0.0 else -1.0
            for index in range(batch_size)
        ]
    )

    targets = MultiTaskTargets(
        event_flag=alternating_event,
        future_low=torch.linspace(98.0, 99.8, steps=batch_size),
        future_high=torch.linspace(101.0, 103.5, steps=batch_size),
        event_start_offset=start_offsets,
        maturity_offset=maturity_offsets,
        confidence_target=(alternating_event * 0.7) + 0.15,
        regime_target=alternating_event.long(),
    )

    losses = compute_multitask_loss(
        event_prediction=event_prediction,
        boundary_prediction=boundary_prediction,
        timing_prediction=timing_prediction,
        confidence_prediction=confidence_prediction,
        regime_prediction=regime_prediction,
        targets=targets,
    )
    metrics = compute_acceptance_metrics(
        event_prediction=event_prediction,
        boundary_prediction=boundary_prediction,
        timing_prediction=timing_prediction,
        confidence_prediction=confidence_prediction,
        regime_prediction=regime_prediction,
        targets=targets,
    )

    losses.total_loss.backward()

    assert losses.total_loss.item() > 0.0
    assert metrics.event_brier >= 0.0
    assert metrics.boundary_mae >= 0.0
    assert metrics.timing_mae is not None
    assert metrics.confidence_brier is not None
    assert metrics.regime_cross_entropy is not None

    assert next(backbone.parameters()).grad is not None
    assert next(event_head.parameters()).grad is not None
    assert next(boundary_head.parameters()).grad is not None
    assert next(timing_head.parameters()).grad is not None
    assert next(confidence_head.parameters()).grad is not None
    assert next(regime_head.parameters()).grad is not None
