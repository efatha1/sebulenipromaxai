"""Unit tests for U6 device selection and fallback."""

from __future__ import annotations

import torch

from models.backbone import Backbone
from models.common import TIMEFRAMES, select_device


def test_select_device_cuda_falls_back_to_cpu_when_unavailable() -> None:
    device = select_device("cuda")
    assert device.type == "cpu"


def test_backbone_runs_on_cpu_when_cuda_unavailable() -> None:
    feature_dim = 4
    batch = 1
    lookback = 2
    windows = {tf: torch.zeros(batch, lookback, feature_dim) for tf in TIMEFRAMES}
    model = Backbone(
        feature_dim=feature_dim,
        seed=7,
        device_preference="cuda",
        allow_nondeterministic=False,
    )
    out = model(windows)
    assert out.fused_latent.device.type == "cpu"

