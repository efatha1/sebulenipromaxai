"""U7 future-boundary prediction head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class BoundaryHeadError(ValueError):
    """Raised when the boundary head receives invalid inputs."""


@dataclass(frozen=True)
class BoundaryPrediction:
    """Typed future-boundary output for inference and offline evaluation."""

    lower_delta: torch.Tensor
    upper_delta: torch.Tensor
    future_low: torch.Tensor
    future_high: torch.Tensor


class BoundaryHead(nn.Module):
    """Predict the lowest and highest reachable prices within the horizon."""

    def __init__(self, *, latent_dim: int) -> None:
        """Initialize the boundary head.

        Args:
            latent_dim: Width of the shared latent vector.

        Raises:
            BoundaryHeadError: If the latent dimension is invalid.
        """
        super().__init__()
        if latent_dim <= 0:
            raise BoundaryHeadError("latent_dim must be positive.")
        self._latent_dim = int(latent_dim)
        self._projection = nn.Linear(self._latent_dim, 2)

    @property
    def latent_dim(self) -> int:
        """Return the configured latent dimension."""
        return self._latent_dim

    def forward(self, latent: torch.Tensor, reference_close: torch.Tensor) -> BoundaryPrediction:
        """Project the latent state to ordered future price boundaries.

        Args:
            latent: Tensor of shape ``(batch, latent_dim)``.
            reference_close: Tensor of shape ``(batch,)`` or ``(batch, 1)``.

        Returns:
            Typed future-boundary output.

        Raises:
            BoundaryHeadError: If inputs are invalid.
        """
        latent_2d = _validate_latent(latent=latent, latent_dim=self._latent_dim)
        ref_close = _validate_reference_close(reference_close=reference_close, batch_size=latent_2d.shape[0])
        ref_close = ref_close.to(device=latent_2d.device, dtype=torch.float32)

        params = self._projection(latent_2d.float())
        center_delta = params[:, 0]
        half_span = F.softplus(params[:, 1])

        future_low = ref_close + center_delta - half_span
        future_high = ref_close + center_delta + half_span
        return BoundaryPrediction(
            lower_delta=future_low - ref_close,
            upper_delta=future_high - ref_close,
            future_low=future_low,
            future_high=future_high,
        )


def predict_boundaries(
    head: BoundaryHead,
    latent: torch.Tensor,
    reference_close: torch.Tensor,
) -> BoundaryPrediction:
    """Run the boundary head with a stable functional interface.

    Args:
        head: Configured boundary head.
        latent: Shared latent tensor of shape ``(batch, latent_dim)``.
        reference_close: Reference close tensor used to reconstruct prices.

    Returns:
        Typed future-boundary output.
    """
    return head(latent, reference_close)


def _validate_latent(*, latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if not isinstance(latent, torch.Tensor):
        raise BoundaryHeadError("latent must be a torch.Tensor.")
    if latent.ndim != 2:
        raise BoundaryHeadError("latent must have shape (batch, latent_dim).")
    if latent.shape[1] != latent_dim:
        raise BoundaryHeadError(
            f"latent feature dimension mismatch: expected {latent_dim}, got {latent.shape[1]}."
        )
    if latent.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise BoundaryHeadError("latent must have a floating-point dtype.")
    if not torch.isfinite(latent).all():
        raise BoundaryHeadError("latent must contain only finite values.")
    return latent


def _validate_reference_close(*, reference_close: torch.Tensor, batch_size: int) -> torch.Tensor:
    if not isinstance(reference_close, torch.Tensor):
        raise BoundaryHeadError("reference_close must be a torch.Tensor.")
    if reference_close.ndim == 2 and reference_close.shape[1] == 1:
        reference_close = reference_close.squeeze(-1)
    if reference_close.ndim != 1:
        raise BoundaryHeadError("reference_close must have shape (batch,) or (batch, 1).")
    if reference_close.shape[0] != batch_size:
        raise BoundaryHeadError(
            f"reference_close batch mismatch: expected {batch_size}, got {reference_close.shape[0]}."
        )
    if reference_close.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise BoundaryHeadError("reference_close must have a floating-point dtype.")
    if not torch.isfinite(reference_close).all():
        raise BoundaryHeadError("reference_close must contain only finite values.")
    return reference_close.float()
