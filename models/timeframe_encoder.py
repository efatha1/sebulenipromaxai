"""Per-timeframe encoder modules (U6)."""

from __future__ import annotations

import logging

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class TimeframeEncoderError(ValueError):
    """Raised when timeframe encoding fails."""


class TimeframeEncoder(nn.Module):
    """Deterministic per-timeframe encoder."""

    def __init__(self, *, feature_dim: int, model_dim: int, latent_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0 or model_dim <= 0 or latent_dim <= 0:
            raise TimeframeEncoderError("feature_dim, model_dim, and latent_dim must be positive.")
        self._feature_dim = int(feature_dim)
        self._model_dim = int(model_dim)
        self._latent_dim = int(latent_dim)

        self._in_norm = nn.LayerNorm(self._feature_dim)
        self._proj = nn.Linear(self._feature_dim, self._model_dim)
        self._act = nn.GELU()
        self._out_norm = nn.LayerNorm(self._model_dim)
        self._latent = nn.Linear(self._model_dim, self._latent_dim)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        if window.ndim != 3:
            raise TimeframeEncoderError("window must have shape (batch, lookback, feature_dim).")
        if window.shape[-1] != self._feature_dim:
            raise TimeframeEncoderError(
                f"window feature_dim mismatch: expected={self._feature_dim} got={int(window.shape[-1])}"
            )
        x = self._in_norm(window)
        x = self._proj(x)
        x = self._act(x)
        x = self._out_norm(x)
        pooled = x.mean(dim=1)
        latent = self._latent(pooled)
        return latent


def encode_timeframe(encoder: TimeframeEncoder, window: torch.Tensor) -> torch.Tensor:
    """Encode a timeframe window into a latent vector."""
    return encoder(window)

