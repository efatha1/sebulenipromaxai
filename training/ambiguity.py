"""Ambiguity handling for U4 label generation."""

from __future__ import annotations

import logging

import pandas as pd

from training.calendar import _build_session_windows, _compute_session_mask
from training.config_schema import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class AmbiguityError(ValueError):
    """Raised when ambiguity classification fails."""


def classify_ambiguity(end_ts: pd.DatetimeIndex, *, horizon_bars: int, config: RuntimeConfig) -> pd.Series:
    """Classify ambiguous label rows for a given horizon.

    Ambiguity is explicit and deterministic:
    - rows are ambiguous if there are insufficient future bars to compute the
      full horizon (tail of series)
    - cadence issues are treated as errors (U2 should have prevented them)
    - session gaps (weekends, etc.) are allowed and do not trigger cadence errors

    Args:
        end_ts: Timestamp index of `1m` bars (timezone-aware).
        horizon_bars: Horizon length in bars.
        config: Validated runtime configuration for session definitions.

    Returns:
        Boolean Series aligned to end_ts indicating ambiguous rows.

    Raises:
        AmbiguityError: If end_ts is invalid or cadence is not `1m` within sessions.
    """
    if horizon_bars <= 0:
        raise AmbiguityError("horizon_bars must be positive.")
    if end_ts.tz is None:
        raise AmbiguityError("end_ts must be timezone-aware.")
    if len(end_ts) == 0:
        raise AmbiguityError("end_ts must not be empty.")

    # Build session mask to identify valid trading periods
    session_windows = _build_session_windows(config)
    session_mask = _compute_session_mask(end_ts, session_windows)

    # Check cadence only within active sessions, not across session boundaries
    diffs = end_ts.to_series().diff().dropna()
    if len(diffs) > 0:
        # Only check cadence for consecutive bars that are both in active sessions
        in_session = session_mask.to_numpy()
        
        # Detect weekend boundary crossings (Friday -> Sunday)
        # This allows the expected 48-hour gap from Friday 16:59 to Sunday 17:00
        prev_weekday = end_ts[:-1].weekday
        curr_weekday = end_ts[1:].weekday
        weekend_boundary = (prev_weekday == 4) & (curr_weekday == 6)
        
        # Only check cadence for gaps that don't cross weekend boundaries
        valid_cadence_mask = pd.Series(in_session[:-1] & in_session[1:] & ~weekend_boundary, index=diffs.index)
        diffs_to_check = diffs[valid_cadence_mask]
        
        if len(diffs_to_check) > 0 and (diffs_to_check != pd.Timedelta(minutes=1)).any():
            # Debug: find the problematic timestamps
            bad_diffs = diffs_to_check[diffs_to_check != pd.Timedelta(minutes=1)]
            print(f"DEBUG: Found {len(bad_diffs)} cadence issues within active sessions")
            for i, (ts, diff) in enumerate(bad_diffs.head(5).items()):
                ts_idx = end_ts.get_loc(ts)
                print(f"  Gap {i+1}: {diff} (expected 1 minute)")
                print(f"    Previous timestamp: {end_ts[ts_idx-1]}")
                print(f"    Current timestamp: {end_ts[ts_idx]}")
            raise AmbiguityError("end_ts cadence is not 1-minute within active sessions; label generation requires validated 1m bars.")

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

