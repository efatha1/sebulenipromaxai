"""Horizon resolution utilities for U4."""

from __future__ import annotations

from typing import Literal

from training.config_schema import RuntimeConfig


class HorizonError(ValueError):
    """Raised when horizon configuration or mode is invalid."""


HorizonMode = Literal["single", "multi"]


def resolve_horizons(
    config: RuntimeConfig,
    *,
    horizon_mode: HorizonMode,
    horizon_bars: int | None = None,
) -> tuple[int, ...]:
    """Resolve horizons for single-horizon or multi-horizon modes.

    Multi-horizon mode uses the configuration-driven fixed horizon list.
    Single-horizon mode requires the requested horizon to be in the fixed list.

    Args:
        config: Validated runtime configuration.
        horizon_mode: 'single' or 'multi'.
        horizon_bars: Horizon bars when in single-horizon mode.

    Returns:
        A tuple of horizon bars (deterministic order).

    Raises:
        HorizonError: If mode is invalid or the single horizon is missing or
            not in the fixed configured list.
    """
    fixed = tuple(int(h) for h in config.labeling.horizon_bars)
    if not fixed:
        raise HorizonError("config.labeling.horizon_bars must not be empty.")

    if horizon_mode == "multi":
        return fixed

    if horizon_mode != "single":
        raise HorizonError(f"Unsupported horizon_mode: {horizon_mode}")

    if horizon_bars is None:
        raise HorizonError("horizon_bars is required in single-horizon mode.")
    if horizon_bars not in fixed:
        raise HorizonError(
            "Single-horizon mode requires horizon_bars to be in the config-driven fixed horizon list. "
            f"horizon_bars={horizon_bars} allowed={fixed}"
        )
    return (int(horizon_bars),)

