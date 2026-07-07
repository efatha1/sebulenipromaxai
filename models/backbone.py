"""Shared multi-timeframe backbone (U6)."""

from __future__ import annotations

import logging

import torch
from torch import nn

from models.common import TIMEFRAMES, BackboneOutput, configure_determinism, select_device
from models.fusion import FusionBlock
from models.timeframe_encoder import TimeframeEncoder, encode_timeframe

LOGGER = logging.getLogger(__name__)


class BackboneError(ValueError):
    """Raised when backbone execution fails."""


class Backbone(nn.Module):
    """Shared backbone that encodes and fuses multi-timeframe feature windows."""

    def __init__(
        self,
        *,
        feature_dim: int,
        seed: int,
        device_preference: str,
        allow_nondeterministic: bool,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise BackboneError("feature_dim must be positive.")

        self._feature_dim = int(feature_dim)
        self._latent_dim = int(feature_dim)
        self._model_dim = int(feature_dim)

        self._device = select_device(device_preference)
        configure_determinism(seed, allow_nondeterministic=allow_nondeterministic)

        self.encoders = nn.ModuleDict(
            {
                tf: TimeframeEncoder(
                    feature_dim=self._feature_dim,
                    model_dim=self._model_dim,
                    latent_dim=self._latent_dim,
                )
                for tf in TIMEFRAMES
            }
        )
        self.fusion = FusionBlock(latent_dim=self._latent_dim)

        self.to(self._device)

        LOGGER.info(
            "initialized_backbone",
            extra={
                "event": "initialized_backbone",
                "feature_dim": self._feature_dim,
                "latent_dim": self._latent_dim,
                "device": str(self._device),
            },
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, windows_by_timeframe: dict[str, torch.Tensor]) -> BackboneOutput:
        if set(windows_by_timeframe.keys()) != set(TIMEFRAMES):
            raise BackboneError(f"windows_by_timeframe must contain exactly {TIMEFRAMES}.")
        if "1m" not in windows_by_timeframe or "5m" not in windows_by_timeframe:
            raise BackboneError("Both 1m and 5m must participate directly in the backbone hierarchy.")

        per_tf: dict[str, torch.Tensor] = {}
        for timeframe in TIMEFRAMES:
            window = windows_by_timeframe[timeframe].to(self._device)
            if window.dtype not in (torch.float32, torch.float64):
                raise BackboneError("Input windows must be float tensors.")
            latent = encode_timeframe(self.encoders[timeframe], window.float())
            per_tf[timeframe] = latent

        fused_latent = self.fusion(per_tf)

        LOGGER.info(
            "backbone_forward",
            extra={
                "event": "backbone_forward",
                "batch_size": int(fused_latent.shape[0]),
                "latent_dim": int(fused_latent.shape[1]),
                "device": str(fused_latent.device),
            },
        )
        return BackboneOutput(fused_latent=fused_latent, per_timeframe_latents=per_tf)

