"""Split helpers for U5 walk-forward evaluation."""

from __future__ import annotations

import pandas as pd


class SplitError(ValueError):
    """Raised when split helpers fail."""


def ensure_sorted_unique_reference_ts(reference_ts: pd.Series) -> pd.DatetimeIndex:
    """Normalize a reference_ts series into a sorted unique DatetimeIndex.

    Args:
        reference_ts: Series of timestamps.

    Returns:
        Sorted timezone-aware DatetimeIndex.

    Raises:
        SplitError: If timestamps are invalid.
    """
    parsed = pd.to_datetime(reference_ts, errors="raise")
    index = pd.DatetimeIndex(parsed)
    if index.tz is None:
        raise SplitError("reference_ts must be timezone-aware.")
    if index.has_duplicates:
        raise SplitError("reference_ts must not contain duplicates.")
    if not index.is_monotonic_increasing:
        index = index.sort_values()
    if not index.is_monotonic_increasing:
        raise SplitError("reference_ts must be monotonically increasing.")
    return index

