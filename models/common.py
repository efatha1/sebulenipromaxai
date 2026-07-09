"""Common utilities and contracts for models (U6)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import torch

LOGGER = logging.getLogger(__name__)

TIMEFRAMES: Final[tuple[str, ...]] = ("1m", "5m", "15m", "1h", "4h", "1d")


class ModelError(ValueError):
    """Raised when model utilities fail."""


@dataclass(frozen=True)
class BackboneOutput:
    """Backbone output contract."""

    fused_latent: torch.Tensor
    per_timeframe_latents: dict[str, torch.Tensor]


def select_device(device_preference: str) -> torch.device:
    """Select execution device with CPU-first fallback behavior.

    Args:
        device_preference: 'cpu' or 'cuda'.

    Returns:
        Selected torch.device.
    """
    preference = device_preference.strip().lower()
    if preference not in {"cpu", "cuda"}:
        raise ModelError(f"Unsupported device_preference: {device_preference}")

    if preference == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            LOGGER.info("selected_device", extra={"event": "selected_device", "device": "cuda"})
            return device
        LOGGER.info(
            "cuda_unavailable_fallback_cpu",
            extra={"event": "cuda_unavailable_fallback_cpu", "device": "cpu"},
        )

    return torch.device("cpu")


def configure_determinism(seed: int, *, allow_nondeterministic: bool) -> None:
    """Configure determinism and random seeds.

    Args:
        seed: Random seed.
        allow_nondeterministic: Whether to allow nondeterministic operations.
    """
    if seed <= 0:
        raise ModelError("seed must be positive.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(not allow_nondeterministic)

    LOGGER.info(
        "configured_determinism",
        extra={
            "event": "configured_determinism",
            "seed": int(seed),
            "allow_nondeterministic": bool(allow_nondeterministic),
        },
    )

