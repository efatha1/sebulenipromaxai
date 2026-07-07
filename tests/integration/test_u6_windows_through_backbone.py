"""Integration tests for U6 backbone consuming U3 windows."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from models.backbone import Backbone
from models.common import TIMEFRAMES
from training.windowing import build_windows


def test_u6_backbone_accepts_feature_windows_contract() -> None:
    feature_names = ("f1", "f2", "f3", "f4")
    tz = timezone.utc

    features_by_tf: dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        rows = 10
        end_ts = pd.DatetimeIndex([datetime(2026, 1, 1, 0, i, tzinfo=tz) for i in range(rows)])
        data = np.random.RandomState(0).randn(rows, len(feature_names)).astype("float64")
        frame = pd.DataFrame(data, columns=feature_names, index=end_ts)
        frame.index.name = "end_ts"
        features_by_tf[tf] = frame

    lookbacks = {tf: 3 for tf in TIMEFRAMES}
    windows = build_windows(features_by_tf, lookbacks_by_timeframe=lookbacks)

    windows_by_tf = {tf: torch.tensor(windows[tf].windows, dtype=torch.float32) for tf in TIMEFRAMES}
    batch, _, feature_dim = windows_by_tf["1m"].shape
    model = Backbone(
        feature_dim=int(feature_dim),
        seed=11,
        device_preference="cpu",
        allow_nondeterministic=False,
    )
    out = model(windows_by_tf)
    assert out.fused_latent.shape[0] == batch
    assert out.fused_latent.shape[1] == feature_dim
