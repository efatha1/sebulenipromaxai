"""Unified multi-horizon multi-timeframe prediction heads.

This module implements the unified 18-output prediction heads that replace
the current single-horizon fused prediction approach. The system generates
labels for all 18 combinations (6 timeframes × 3 horizons) independently.
"""

from __future__ import annotations

import torch
from torch import nn


class UnifiedEventHead(nn.Module):
    """Unified event prediction head for all 18 combinations.

    Outputs a single tensor of shape (batch, 18) where each output corresponds
    to a specific (timeframe, horizon) combination in the order:
    1m_h15, 1m_h60, 1m_h120, 5m_h15, 5m_h60, 5m_h120, ..., 1d_h120
    """

    def __init__(self, latent_dim: int, num_outputs: int = 18) -> None:
        """Initialize the unified event head.

        Args:
            latent_dim: Dimension of the fused latent representation.
            num_outputs: Number of output combinations (default 18 for 6×3).
        """
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_outputs)

    def forward(self, fused_latent: torch.Tensor) -> torch.Tensor:
        """Forward pass through the unified event head.

        Args:
            fused_latent: Fused latent representation of shape (batch, latent_dim).

        Returns:
            Event logits of shape (batch, 18).
        """
        return self.linear(fused_latent)


class UnifiedBoundaryHead(nn.Module):
    """Unified boundary prediction head for all 18 combinations.

    Outputs a tensor of shape (batch, 18, 2) where the last dimension
    represents (future_low, future_high) for each combination.
    """

    def __init__(self, latent_dim: int, num_outputs: int = 36) -> None:
        """Initialize the unified boundary head.

        Args:
            latent_dim: Dimension of the fused latent representation.
            num_outputs: Number of output values (default 36 for 18×2).
        """
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_outputs)

    def forward(self, fused_latent: torch.Tensor) -> torch.Tensor:
        """Forward pass through the unified boundary head.

        Args:
            fused_latent: Fused latent representation of shape (batch, latent_dim).

        Returns:
            Boundary predictions of shape (batch, 18, 2) where the last dimension
            is (future_low, future_high).
        """
        output = self.linear(fused_latent)  # (batch, 36)
        # Reshape to (batch, 18, 2) for (future_low, future_high) per combination
        return output.view(-1, 18, 2)


class UnifiedTimingHead(nn.Module):
    """Unified timing prediction head for all 18 combinations with timeframe-aware scaling.

    Outputs a tensor of shape (batch, 18, 2) where the last dimension
    represents (event_start_offset, maturity_offset) for each combination.
    Applies timeframe-aware scaling to account for different bar durations.
    """

    def __init__(self, latent_dim: int, num_outputs: int = 36, max_horizon_bars: int = 120) -> None:
        """Initialize the unified timing head.

        Args:
            latent_dim: Dimension of the fused latent representation.
            num_outputs: Number of output values (default 36 for 18×2).
            max_horizon_bars: Maximum horizon in bars for clamping.
        """
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_outputs)
        self.max_horizon_bars = max_horizon_bars
        # Timeframe-specific scaling factors (1m, 5m, 15m, 1h, 4h, 1d)
        # Each timeframe gets 3 horizons, so scale repeats 3 times
        self.register_buffer(
            "timeframe_scale",
            torch.tensor([1.0, 5.0, 15.0, 60.0, 240.0, 1440.0]).repeat_interleave(3),
        )

    def forward(self, fused_latent: torch.Tensor) -> torch.Tensor:
        """Forward pass through the unified timing head.

        Args:
            fused_latent: Fused latent representation of shape (batch, latent_dim).

        Returns:
            Timing predictions of shape (batch, 18, 2) where the last dimension
            is (event_start_offset, maturity_offset), scaled by timeframe duration.
        """
        output = self.linear(fused_latent)  # (batch, 36)
        output = output.view(-1, 18, 2)  # (batch, 18, 2) for (event_start, maturity)

        # Apply timeframe-aware scaling
        # scale has shape (18,), broadcast to (batch, 18, 2)
        output = output * self.timeframe_scale.unsqueeze(0).unsqueeze(-1)

        # Bound to [0, max_horizon_bars]
        output = torch.clamp(output, 0, self.max_horizon_bars)
        return output
