"""Unit tests for U4 ambiguity handling."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from training.ambiguity import classify_ambiguity


def test_tail_rows_are_ambiguous_when_insufficient_future() -> None:
    end_ts = pd.DatetimeIndex([datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc) for i in range(10)])
    mask = classify_ambiguity(end_ts, horizon_bars=3)

    assert bool(mask.iloc[0]) is False
    assert bool(mask.iloc[6]) is False
    assert bool(mask.iloc[7]) is True
    assert bool(mask.iloc[9]) is True
