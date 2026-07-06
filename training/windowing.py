"""Window building utilities for U3."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class WindowingError(ValueError):
    """Raised when building feature windows fails."""


@dataclass(frozen=True)
class FeatureWindows:
    """Feature windows keyed by reference timestamp."""

    reference_ts: pd.DatetimeIndex
    windows: np.ndarray
    feature_names: tuple[str, ...]


def build_windows(
    features_by_timeframe: dict[str, pd.DataFrame],
    lookbacks_by_timeframe: dict[str, int],
) -> dict[str, FeatureWindows]:
    """Build causal feature windows for each timeframe.

    Windows are strictly trailing: each window ends at its reference timestamp.

    Args:
        features_by_timeframe: Mapping timeframe -> feature frame indexed by `end_ts`.
        lookbacks_by_timeframe: Mapping timeframe -> trailing window length.

    Returns:
        Mapping timeframe -> FeatureWindows.

    Raises:
        WindowingError: If lookbacks are invalid, frames are misaligned, or the
            requested window cannot be built.
    """
    expected = {"1m", "5m", "15m", "1h", "4h", "1d"}
    if set(features_by_timeframe.keys()) != expected:
        raise WindowingError(f"features_by_timeframe must contain exactly {sorted(expected)}.")
    if set(lookbacks_by_timeframe.keys()) != expected:
        raise WindowingError(f"lookbacks_by_timeframe must contain exactly {sorted(expected)}.")

    out: dict[str, FeatureWindows] = {}
    for timeframe, frame in features_by_timeframe.items():
        lookback = lookbacks_by_timeframe[timeframe]
        out[timeframe] = _build_timeframe_windows(frame, timeframe=timeframe, lookback=lookback)

    LOGGER.info(
        "built_feature_windows",
        extra={
            "event": "built_feature_windows",
            "timeframes": sorted(out.keys()),
        },
    )
    return out


def _build_timeframe_windows(frame: pd.DataFrame, *, timeframe: str, lookback: int) -> FeatureWindows:
    if lookback <= 0:
        raise WindowingError(f"lookback must be positive for timeframe={timeframe}.")
    if len(frame) < lookback:
        raise WindowingError(f"Not enough rows to build windows for timeframe={timeframe}.")
    if frame.index.tz is None:
        raise WindowingError(f"Feature frame index must be timezone-aware for timeframe={timeframe}.")

    values = frame.to_numpy(dtype=np.float64)
    if np.isnan(values).any() or np.isinf(values).any():
        raise WindowingError(f"Feature frame contains NaN/inf values for timeframe={timeframe}.")

    n_rows, n_features = values.shape
    n_windows = n_rows - lookback + 1
    windows = np.empty((n_windows, lookback, n_features), dtype=np.float64)
    for i in range(n_windows):
        windows[i] = values[i : i + lookback]

    reference_ts = pd.DatetimeIndex(frame.index[lookback - 1 :])
    if len(reference_ts) != n_windows:
        raise WindowingError(f"Reference timestamp mismatch for timeframe={timeframe}.")

    return FeatureWindows(reference_ts=reference_ts, windows=windows, feature_names=tuple(frame.columns))

