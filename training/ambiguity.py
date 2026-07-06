"""Ambiguity handling for U4 label generation."""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


class AmbiguityError(ValueError):
    """Raised when ambiguity classification fails."""


def classify_ambiguity(end_ts: pd.DatetimeIndex, *, horizon_bars: int) -> pd.Series:
    """Classify ambiguous label rows for a given horizon.

    Ambiguity is explicit and deterministic:
    - rows are ambiguous if there are insufficient future bars to compute the
      full horizon (tail of series)
    - cadence issues are treated as errors (U2 should have prevented them)

    Args:
        end_ts: Timestamp index of `1m` bars (timezone-aware).
        horizon_bars: Horizon length in bars.

    Returns:
        Boolean Series aligned to end_ts indicating ambiguous rows.

    Raises:
        AmbiguityError: If end_ts is invalid or cadence is not `1m`.
    """
    if horizon_bars <= 0:
        raise AmbiguityError("horizon_bars must be positive.")
    if end_ts.tz is None:
        raise AmbiguityError("end_ts must be timezone-aware.")
    if len(end_ts) == 0:
        raise AmbiguityError("end_ts must not be empty.")

    diffs = end_ts.to_series().diff().dropna()
    if len(diffs) > 0 and (diffs != pd.Timedelta(minutes=1)).any():
        raise AmbiguityError("end_ts cadence is not 1-minute; label generation requires validated 1m bars.")

    n = len(end_ts)
    ambiguous = [False] * n
    for i in range(n):
        if i + horizon_bars >= n:
            ambiguous[i] = True
    out = pd.Series(ambiguous, index=end_ts, dtype=bool)

    LOGGER.info(
        "classified_ambiguity",
        extra={
            "event": "classified_ambiguity",
            "horizon_bars": int(horizon_bars),
            "ambiguous_count": int(out.sum()),
            "row_count": int(len(out)),
        },
    )
    return out

