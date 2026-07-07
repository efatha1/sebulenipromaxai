"""U7 timing prediction head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class TimingHeadError(ValueError):
    """Raised when the timing head receives invalid inputs."""


@dataclass(frozen=True)
class TimingPrediction:
    """Typed timing output preserving event-start and maturity semantics."""

    event_start_offset: torch.Tensor
    maturity_offset: torch.Tensor


class TimingHead(nn.Module):
    """Predict bounded event-start and maturity offsets."""

    def __init__(self, *, latent_dim: int, max_horizon_bars: int) -> None:
        """Initialize the timing head.

        Args:
            latent_dim: Width of the shared latent vector.
            max_horizon_bars: Maximum supported future horizon in bars.

        Raises:
            TimingHeadError: If configuration is invalid.
        """
        super().__init__()
        if latent_dim <= 0:
            raise TimingHeadError("latent_dim must be positive.")
        if max_horizon_bars <= 0:
            raise TimingHeadError("max_horizon_bars must be positive.")
        self._latent_dim = int(latent_dim)
        self._max_horizon_bars = int(max_horizon_bars)
        self._projection = nn.Linear(self._latent_dim, 2)

    @property
    def latent_dim(self) -> int:
        """Return the configured latent dimension."""
        return self._latent_dim

    @property
    def max_horizon_bars(self) -> int:
        """Return the configured maximum horizon."""
        return self._max_horizon_bars

    def forward(self, latent: torch.Tensor) -> TimingPrediction:
        """Project the latent state to bounded timing offsets.

        Args:
            latent: Tensor of shape ``(batch, latent_dim)``.

        Returns:
            Typed timing output whose offsets lie in ``[1, max_horizon_bars]``
            and satisfy ``maturity_offset >= event_start_offset``.

        Raises:
            TimingHeadError: If the latent tensor is invalid.
        """
        latent_2d = _validate_latent(latent=latent, latent_dim=self._latent_dim)
        params = self._projection(latent_2d.float())

        start_unit = torch.sigmoid(params[:, 0])
        gap_unit = torch.sigmoid(params[:, 1])

        event_start_offset = _map_unit_interval_to_offsets(
            unit_values=start_unit,
            max_horizon_bars=self._max_horizon_bars,
        )
        maturity_gap = gap_unit * float(max(0, self._max_horizon_bars - 1))
        maturity_offset = torch.clamp(event_start_offset + maturity_gap, max=float(self._max_horizon_bars))
        return TimingPrediction(
            event_start_offset=event_start_offset,
            maturity_offset=maturity_offset,
        )


def predict_timing(head: TimingHead, latent: torch.Tensor) -> TimingPrediction:
    """Run the timing head with a stable functional interface.

    Args:
        head: Configured timing head.
        latent: Shared latent tensor of shape ``(batch, latent_dim)``.

    Returns:
        Typed timing output.
    """
    return head(latent)


def _validate_latent(*, latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if not isinstance(latent, torch.Tensor):
        raise TimingHeadError("latent must be a torch.Tensor.")
    if latent.ndim != 2:
        raise TimingHeadError("latent must have shape (batch, latent_dim).")
    if latent.shape[1] != latent_dim:
        raise TimingHeadError(
            f"latent feature dimension mismatch: expected {latent_dim}, got {latent.shape[1]}."
        )
    if latent.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise TimingHeadError("latent must have a floating-point dtype.")
    if not torch.isfinite(latent).all():
        raise TimingHeadError("latent must contain only finite values.")
    return latent


def _map_unit_interval_to_offsets(*, unit_values: torch.Tensor, max_horizon_bars: int) -> torch.Tensor:
    if max_horizon_bars == 1:
        return torch.ones_like(unit_values)
    return 1.0 + (unit_values * float(max_horizon_bars - 1))
