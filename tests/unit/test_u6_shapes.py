"""Unit tests for U6 backbone shapes."""

from __future__ import annotations

import torch

from models.backbone import Backbone
from models.common import TIMEFRAMES


def test_backbone_output_shapes() -> None:
    feature_dim = 8
    batch = 4
    lookback = 5

    windows = {tf: torch.randn(batch, lookback, feature_dim) for tf in TIMEFRAMES}
    model = Backbone(
        feature_dim=feature_dim,
        seed=123,
        device_preference="cpu",
        allow_nondeterministic=False,
    )

    out = model(windows)
    assert out.fused_latent.shape == (batch, feature_dim)
    assert set(out.per_timeframe_latents.keys()) == set(TIMEFRAMES)
    for tf in TIMEFRAMES:
        assert out.per_timeframe_latents[tf].shape == (batch, feature_dim)

