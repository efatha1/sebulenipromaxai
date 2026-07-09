"""Unit tests for preprocessing cross-timeframe timestamp alignment.

These tests cover the alignment policy implemented in `Preprocessing.py`:
- `1m` requires exact equality between label timestamps and window reference timestamps.
- Higher timeframes map each label timestamp to the latest window reference timestamp
  at or before the label timestamp (causal, no future leakage).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from Preprocessing import _compute_common_history_start, _resolve_lookbacks_by_timeframe
from training.config_loader import validate_config


def _ns(ts: pd.DatetimeIndex) -> np.ndarray:
    """Convert timezone-aware timestamps to int64 ns since epoch."""
    if ts.tz is None:
        raise ValueError("test timestamps must be timezone-aware")
    return ts.view("int64")


def test_higher_timeframe_aligns_to_latest_closed_bar() -> None:
    # Higher timeframe closes: 00:05, 00:10, 00:15
    tf_ref = pd.DatetimeIndex(
        [
            datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
        ]
    )
    # Labels occur each minute: 00:11 should align to 00:10, 00:15 aligns to itself.
    labels = pd.DatetimeIndex(
        [
            datetime(2026, 1, 1, 0, 11, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
        ]
    )

    positions = np.searchsorted(_ns(tf_ref), _ns(labels), side="right") - 1
    assert positions.tolist() == [1, 2]
    matched = _ns(tf_ref)[positions]
    assert (matched <= _ns(labels)).all()


def test_higher_timeframe_rejects_label_before_first_reference() -> None:
    tf_ref = pd.DatetimeIndex([datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)])
    labels = pd.DatetimeIndex([datetime(2026, 1, 1, 0, 4, tzinfo=timezone.utc)])
    positions = np.searchsorted(_ns(tf_ref), _ns(labels), side="right") - 1
    assert positions.tolist() == [-1]


def test_1m_exact_match_policy_behavior() -> None:
    # This mirrors the 1m "exact match" requirement: searchsorted-left + equality.
    tf_ref = pd.DatetimeIndex(
        [
            datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc),
        ]
    )
    labels_ok = pd.DatetimeIndex([datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)])
    pos_ok = np.searchsorted(_ns(tf_ref), _ns(labels_ok), side="left")
    assert _ns(tf_ref)[pos_ok][0] == _ns(labels_ok)[0]

    labels_bad = pd.DatetimeIndex([datetime(2026, 1, 1, 0, 2, 30, tzinfo=timezone.utc)])
    pos_bad = np.searchsorted(_ns(tf_ref), _ns(labels_bad), side="left")
    # searchsorted-left returns the next >= timestamp (00:03) rather than an exact match.
    assert _ns(tf_ref)[pos_bad][0] != _ns(labels_bad)[0]


def test_common_history_start_is_max_across_timeframes() -> None:
    # 1m has enough history earlier; 5m needs more warmup.
    idx_1m = pd.date_range("2026-01-01", periods=200, freq="min", tz="UTC")
    idx_5m = pd.date_range("2026-01-01 00:05", periods=200, freq="5min", tz="UTC")
    idx_15m = pd.date_range("2026-01-01 00:15", periods=200, freq="15min", tz="UTC")
    idx_1h = pd.date_range("2026-01-01 01:00", periods=200, freq="1h", tz="UTC")
    idx_4h = pd.date_range("2026-01-01 04:00", periods=200, freq="4h", tz="UTC")
    idx_1d = pd.date_range("2026-01-02", periods=200, freq="1d", tz="UTC")

    features_by_tf = {
        "1m": pd.DataFrame(index=idx_1m, data={"f": 0.0}),
        "5m": pd.DataFrame(index=idx_5m, data={"f": 0.0}),
        "15m": pd.DataFrame(index=idx_15m, data={"f": 0.0}),
        "1h": pd.DataFrame(index=idx_1h, data={"f": 0.0}),
        "4h": pd.DataFrame(index=idx_4h, data={"f": 0.0}),
        "1d": pd.DataFrame(index=idx_1d, data={"f": 0.0}),
    }
    lookbacks_by_tf = {tf: 3 for tf in features_by_tf}
    common = _compute_common_history_start(features_by_tf=features_by_tf, lookbacks_by_tf=lookbacks_by_tf)
    assert common == max(pd.Timestamp(idx[2]) for idx in (idx_1m, idx_5m, idx_15m, idx_1h, idx_4h, idx_1d))


def test_resolve_lookbacks_by_timeframe_prefers_config_mapping() -> None:
    config = validate_config(
        {
            "instrument": {"instrument_id": "TEST_INSTRUMENT"},
            "data_source": {"ohlc_path": "data/test.csv"},
            "time": {
                "source_timezone": "UTC",
                "runtime_timezone": "UTC",
                "daily_close": {"time": "17:00", "timezone": "America/New_York"},
            },
            "sessions": {
                "calendar_name": "default",
                "weekend_policy": "include",
                "holiday_policy": "include",
                "definitions": [{"name": "primary", "start": "00:00", "end": "23:59"}],
            },
            "resampling": {"base_timeframe": "1m", "target_timeframes": ["5m", "15m", "1h", "4h", "1d"]},
            "features": {
                "enabled_features": ["returns", "ranges", "wick_body_ratios"],
                "deterministic_derived_features": ["returns", "ranges", "wick_body_ratios"],
            },
            "labeling": {"thresholds": [10.0], "horizon_bars": [15]},
            "walk_forward": {"train_bars": 1000, "validation_bars": 200, "test_bars": 200, "step_bars": 200},
            "training": {
                "random_seed": 1,
                "batch_size": 1,
                "learning_rate": 0.001,
                "max_epochs": 1,
                "device_preference": "cpu",
                "allow_nondeterministic": False,
            },
            "preprocessing": {
                "lookbacks_by_timeframe": {
                    "1m": 90,
                    "5m": 60,
                    "15m": 48,
                    "1h": 24,
                    "4h": 16,
                    "1d": 10,
                }
            },
            "retrieval": {"top_k_analogs": 1},
            "api": {"host": "127.0.0.1", "port": 8000},
            "reporting": {"output_dir": "artifacts"},
            "schedules": {"reporting_cron": "0 9 * * *", "retraining_review_cron": "0 0 * * 1"},
        }
    )

    resolved = _resolve_lookbacks_by_timeframe(config=config, fallback_lookback=123)
    assert resolved == {"1m": 90, "5m": 60, "15m": 48, "1h": 24, "4h": 16, "1d": 10}
