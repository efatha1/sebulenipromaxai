"""U7 confidence prediction head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ConfidenceHeadError(ValueError):
    """Raised when the confidence head receives invalid inputs."""


@dataclass(frozen=True)
class ConfidencePrediction:
    """Typed confidence output for downstream gating and reporting."""

    logits: torch.Tensor
    confidence: torch.Tensor


class ConfidenceHead(nn.Module):
    """Predict confidence from a shared latent state."""

    def __init__(self, *, latent_dim: int) -> None:
        """Initialize the confidence head.

        Args:
            latent_dim: Width of the shared latent vector.

        Raises:
            ConfidenceHeadError: If the latent dimension is invalid.
        """
        super().__init__()
        if latent_dim <= 0:
            raise ConfidenceHeadError("latent_dim must be positive.")
        self._latent_dim = int(latent_dim)
        self._projection = nn.Linear(self._latent_dim, 1)

    @property
    def latent_dim(self) -> int:
        """Return the configured latent dimension."""
        return self._latent_dim

    def forward(self, latent: torch.Tensor) -> ConfidencePrediction:
        """Project the latent state to confidence logits and scores.

        Args:
            latent: Tensor of shape ``(batch, latent_dim)``.

        Returns:
            Typed confidence output.

        Raises:
            ConfidenceHeadError: If the latent tensor is invalid.
        """
        latent_2d = _validate_latent(latent=latent, latent_dim=self._latent_dim)
        logits = self._projection(latent_2d.float()).squeeze(-1)
        return ConfidencePrediction(
            logits=logits,
            confidence=confidence_from_logit(logits),
        )


def predict_confidence(head: ConfidenceHead, latent: torch.Tensor) -> ConfidencePrediction:
    """Run the confidence head with a stable functional interface.

    Args:
        head: Configured confidence head.
        latent: Shared latent tensor of shape ``(batch, latent_dim)``.

    Returns:
        Typed confidence output.
    """
    return head(latent)


def confidence_from_logit(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to a bounded confidence score.

    Args:
        logits: Confidence logits.

    Returns:
        Tensor with values in ``[0, 1]``.

    Raises:
        ConfidenceHeadError: If logits are invalid.
    """
    if not isinstance(logits, torch.Tensor):
        raise ConfidenceHeadError("logits must be a torch.Tensor.")
    if logits.ndim != 1:
        raise ConfidenceHeadError("logits must have shape (batch,).")
    if logits.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ConfidenceHeadError("logits must have a floating-point dtype.")
    if not torch.isfinite(logits).all():
        raise ConfidenceHeadError("logits must contain only finite values.")
    return torch.sigmoid(logits)


def _validate_latent(*, latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if not isinstance(latent, torch.Tensor):
        raise ConfidenceHeadError("latent must be a torch.Tensor.")
    if latent.ndim != 2:
        raise ConfidenceHeadError("latent must have shape (batch, latent_dim).")
    if latent.shape[1] != latent_dim:
        raise ConfidenceHeadError(
            f"latent feature dimension mismatch: expected {latent_dim}, got {latent.shape[1]}."
        )
    if latent.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ConfidenceHeadError("latent must have a floating-point dtype.")
    if not torch.isfinite(latent).all():
        raise ConfidenceHeadError("latent must contain only finite values.")
    return latent
