"""Unit tests for U6 determinism under fixed seed."""

from __future__ import annotations

import torch

from models.backbone import Backbone
from models.common import TIMEFRAMES


def test_backbone_determinism_same_seed_same_output() -> None:
    feature_dim = 8
    batch = 2
    lookback = 4

    torch.manual_seed(999)
    windows = {tf: torch.randn(batch, lookback, feature_dim) for tf in TIMEFRAMES}

    model1 = Backbone(
        feature_dim=feature_dim,
        seed=42,
        device_preference="cpu",
        allow_nondeterministic=False,
    )
    out1 = model1(windows).fused_latent.detach().cpu()

    model2 = Backbone(
        feature_dim=feature_dim,
        seed=42,
        device_preference="cpu",
        allow_nondeterministic=False,
    )
    out2 = model2(windows).fused_latent.detach().cpu()

    assert torch.equal(out1, out2)

