"""U7 regime scoring head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class RegimeHeadError(ValueError):
    """Raised when the regime head receives invalid inputs."""


@dataclass(frozen=True)
class RegimePrediction:
    """Typed regime output for downstream monitoring and routing."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_regime: torch.Tensor


class RegimeHead(nn.Module):
    """Score regime classes from a shared latent state."""

    def __init__(self, *, latent_dim: int, num_regimes: int = 2) -> None:
        """Initialize the regime head.

        Args:
            latent_dim: Width of the shared latent vector.
            num_regimes: Number of supported regime classes.

        Raises:
            RegimeHeadError: If configuration is invalid.
        """
        super().__init__()
        if latent_dim <= 0:
            raise RegimeHeadError("latent_dim must be positive.")
        if num_regimes <= 1:
            raise RegimeHeadError("num_regimes must be greater than 1.")
        self._latent_dim = int(latent_dim)
        self._num_regimes = int(num_regimes)
        self._projection = nn.Linear(self._latent_dim, self._num_regimes)

    @property
    def latent_dim(self) -> int:
        """Return the configured latent dimension."""
        return self._latent_dim

    @property
    def num_regimes(self) -> int:
        """Return the configured number of regime classes."""
        return self._num_regimes

    def forward(self, latent: torch.Tensor) -> RegimePrediction:
        """Project the latent state to regime logits and probabilities.

        Args:
            latent: Tensor of shape ``(batch, latent_dim)``.

        Returns:
            Typed regime prediction output.

        Raises:
            RegimeHeadError: If the latent tensor is invalid.
        """
        latent_2d = _validate_latent(latent=latent, latent_dim=self._latent_dim)
        logits = self._projection(latent_2d.float())
        probabilities = torch.softmax(logits, dim=1)
        predicted_regime = torch.argmax(probabilities, dim=1)
        return RegimePrediction(
            logits=logits,
            probabilities=probabilities,
            predicted_regime=predicted_regime,
        )


def score_regime(head: RegimeHead, latent: torch.Tensor) -> RegimePrediction:
    """Run the regime head with a stable functional interface.

    Args:
        head: Configured regime head.
        latent: Shared latent tensor of shape ``(batch, latent_dim)``.

    Returns:
        Typed regime prediction output.
    """
    return head(latent)


def _validate_latent(*, latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if not isinstance(latent, torch.Tensor):
        raise RegimeHeadError("latent must be a torch.Tensor.")
    if latent.ndim != 2:
        raise RegimeHeadError("latent must have shape (batch, latent_dim).")
    if latent.shape[1] != latent_dim:
        raise RegimeHeadError(
            f"latent feature dimension mismatch: expected {latent_dim}, got {latent.shape[1]}."
        )
    if latent.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise RegimeHeadError("latent must have a floating-point dtype.")
    if not torch.isfinite(latent).all():
        raise RegimeHeadError("latent must contain only finite values.")
    return latent
