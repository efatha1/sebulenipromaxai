"""Cross-timeframe fusion utilities (U6)."""

from __future__ import annotations

import logging

import torch
from torch import nn

from models.common import TIMEFRAMES

LOGGER = logging.getLogger(__name__)


class FusionError(ValueError):
    """Raised when fusion fails."""


class FusionBlock(nn.Module):
    """Deterministic fusion block for per-timeframe latents."""

    def __init__(self, *, latent_dim: int) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise FusionError("latent_dim must be positive.")
        self._latent_dim = int(latent_dim)
        self._in_norm = nn.LayerNorm(self._latent_dim * len(TIMEFRAMES))
        self._proj = nn.Linear(self._latent_dim * len(TIMEFRAMES), self._latent_dim)
        self._out_norm = nn.LayerNorm(self._latent_dim)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, per_timeframe_latents: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = fuse_latents(per_timeframe_latents, latent_dim=self._latent_dim)
        fused = self._in_norm(fused)
        fused = self._proj(fused)
        fused = self._out_norm(fused)
        return fused


def fuse_latents(per_timeframe_latents: dict[str, torch.Tensor], *, latent_dim: int) -> torch.Tensor:
    """Fuse per-timeframe latents into a single vector.

    Args:
        per_timeframe_latents: Mapping timeframe -> latent tensor (batch, latent_dim).
        latent_dim: Expected latent dimension for each timeframe.

    Returns:
        Fused tensor (batch, latent_dim * num_timeframes) before projection.
    """
    required = set(TIMEFRAMES)
    if set(per_timeframe_latents.keys()) != required:
        raise FusionError(f"per_timeframe_latents must contain exactly {TIMEFRAMES}.")
    if "1m" not in per_timeframe_latents or "5m" not in per_timeframe_latents:
        raise FusionError("Both 1m and 5m must participate directly in fusion.")

    tensors: list[torch.Tensor] = []
    batch_size: int | None = None
    for timeframe in TIMEFRAMES:
        latent = per_timeframe_latents[timeframe]
        if latent.ndim != 2 or latent.shape[1] != latent_dim:
            raise FusionError(f"Latent for timeframe={timeframe} must have shape (batch, {latent_dim}).")
        if batch_size is None:
            batch_size = int(latent.shape[0])
        elif int(latent.shape[0]) != batch_size:
            raise FusionError("All timeframe latents must have the same batch size.")
        tensors.append(latent)

    return torch.cat(tensors, dim=1)

