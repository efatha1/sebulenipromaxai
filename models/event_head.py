"""U7 event prediction head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class EventHeadError(ValueError):
    """Raised when the event head receives invalid inputs."""


@dataclass(frozen=True)
class EventPrediction:
    """Typed event-head output for offline evaluation and inference."""

    logits: torch.Tensor
    probabilities: torch.Tensor


class EventHead(nn.Module):
    """Predict event probability from a shared latent state."""

    def __init__(self, *, latent_dim: int) -> None:
        """Initialize the event head.

        Args:
            latent_dim: Width of the shared latent vector.

        Raises:
            EventHeadError: If the latent dimension is invalid.
        """
        super().__init__()
        if latent_dim <= 0:
            raise EventHeadError("latent_dim must be positive.")
        self._latent_dim = int(latent_dim)
        self._projection = nn.Linear(self._latent_dim, 1)

    @property
    def latent_dim(self) -> int:
        """Return the configured latent dimension."""
        return self._latent_dim

    def forward(self, latent: torch.Tensor) -> EventPrediction:
        """Project the latent state to event logits and probabilities.

        Args:
            latent: Tensor of shape ``(batch, latent_dim)``.

        Returns:
            Typed event prediction output.

        Raises:
            EventHeadError: If the latent tensor is invalid.
        """
        latent_2d = _validate_latent(latent=latent, latent_dim=self._latent_dim)
        logits = self._projection(latent_2d.float()).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return EventPrediction(logits=logits, probabilities=probabilities)


def predict_event(head: EventHead, latent: torch.Tensor) -> EventPrediction:
    """Run the event head with a stable functional interface.

    Args:
        head: Configured event head.
        latent: Shared latent tensor of shape ``(batch, latent_dim)``.

    Returns:
        Typed event prediction output.
    """
    return head(latent)


def _validate_latent(*, latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if not isinstance(latent, torch.Tensor):
        raise EventHeadError("latent must be a torch.Tensor.")
    if latent.ndim != 2:
        raise EventHeadError("latent must have shape (batch, latent_dim).")
    if latent.shape[1] != latent_dim:
        raise EventHeadError(
            f"latent feature dimension mismatch: expected {latent_dim}, got {latent.shape[1]}."
        )
    if latent.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise EventHeadError("latent must have a floating-point dtype.")
    if not torch.isfinite(latent).all():
        raise EventHeadError("latent must contain only finite values.")
    return latent
