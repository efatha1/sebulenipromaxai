"""U4 deterministic label generation and horizon engine."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from training.ambiguity import classify_ambiguity
from training.config_schema import MODELED_TIMEFRAMES, RuntimeConfig
from training.horizons import HorizonMode, resolve_horizons

LOGGER = logging.getLogger(__name__)


class LabelingError(ValueError):
    """Raised when label generation fails."""


def generate_labels(
    bars_1m: pd.DataFrame,
    config: RuntimeConfig,
    *,
    horizon_mode: HorizonMode,
    horizon_bars: int | None = None,
    thresholds: tuple[float, ...] | None = None,
) -> dict[tuple[int, float], pd.DataFrame]:
    """Generate labels for each (horizon, threshold) combination.

    Semantics are defined by the SRS:
    - reference price is the latest closed-bar close at reference_ts
    - event occurs if future highest high OR future lowest low produces an
      absolute excursion crossing the threshold within the horizon
    - event start is the first future bar where the threshold crossing occurs
    - maturity is the point of maximum excursion within the horizon
    - future_low/future_high are min/max reachable prices within the horizon

    Args:
        bars_1m: Validated 1m bars with columns: end_ts, open, high, low, close.
        config: Validated runtime configuration.
        horizon_mode: 'single' or 'multi'.
        horizon_bars: Horizon in bars for single-horizon mode.
        thresholds: Optional override thresholds. If omitted, uses config-driven
            fixed threshold list.

    Returns:
        Mapping (horizon_bars, threshold) -> label DataFrame with columns:
        reference_ts, horizon_bars, threshold, event_flag, event_start_offset,
        maturity_offset, future_low, future_high, ambiguous.

    Raises:
        LabelingError: If inputs are invalid or labels cannot be computed.
    """
    bars = _standardize_1m_bars(bars_1m)

    horizons = resolve_horizons(config, horizon_mode=horizon_mode, horizon_bars=horizon_bars)
    thresholds_to_use = thresholds if thresholds is not None else tuple(float(t) for t in config.labeling.thresholds)
    if not thresholds_to_use:
        raise LabelingError("Threshold list must not be empty.")

    out: dict[tuple[int, float], pd.DataFrame] = {}
    for horizon in horizons:
        ambiguity_mask = classify_ambiguity(pd.DatetimeIndex(bars.index), horizon_bars=int(horizon), config=config)
        for threshold in thresholds_to_use:
            labels = _labels_for_horizon_threshold(
                bars,
                horizon_bars=int(horizon),
                threshold=float(threshold),
                ambiguous_mask=ambiguity_mask,
            )
            out[(int(horizon), float(threshold))] = labels

    LOGGER.info(
        "generated_labels",
        extra={
            "event": "generated_labels",
            "horizon_mode": horizon_mode,
            "horizons": list(horizons),
            "threshold_count": len(thresholds_to_use),
            "row_count": int(len(bars)),
        },
    )
    return out


def generate_labels_multi_timeframe(
    bars_by_timeframe: dict[str, pd.DataFrame],
    config: RuntimeConfig,
    *,
    horizon_mode: HorizonMode,
    horizon_bars: int | None = None,
    thresholds: tuple[float, ...] | None = None,
) -> dict[tuple[str, int, float], pd.DataFrame]:
    """Generate labels for each (timeframe, horizon, threshold) combination.

    For each timeframe in MODELED_TIMEFRAMES, generates labels using that timeframe's
    own bars. Higher timeframes (5m, 15m, 1h, 4h, 1d) use their resampled bars for
    label generation. Ambiguity classification uses horizon_bars in 1-minute equivalents,
    converted per timeframe.

    Args:
        bars_by_timeframe: Mapping of timeframe -> DataFrame with columns:
            end_ts, open, high, low, close. Must include all MODELED_TIMEFRAMES.
        config: Validated runtime configuration.
        horizon_mode: 'single' or 'multi'.
        horizon_bars: Horizon in bars for single-horizon mode.
        thresholds: Optional override thresholds. If omitted, uses config-driven
            fixed threshold list.

    Returns:
        Mapping (timeframe, horizon_bars, threshold) -> label DataFrame with columns:
        reference_ts, horizon_bars, threshold, event_flag, event_start_offset,
        maturity_offset, future_low, future_high, ambiguous.

    Raises:
        LabelingError: If inputs are invalid or labels cannot be computed.
    """
    # Validate that all required timeframes are present
    missing_timeframes = set(MODELED_TIMEFRAMES) - set(bars_by_timeframe.keys())
    if missing_timeframes:
        raise LabelingError(
            f"bars_by_timeframe missing required timeframes: {sorted(missing_timeframes)}"
        )

    horizons = resolve_horizons(config, horizon_mode=horizon_mode, horizon_bars=horizon_bars)
    thresholds_to_use = thresholds if thresholds is not None else tuple(float(t) for t in config.labeling.thresholds)
    if not thresholds_to_use:
        raise LabelingError("Threshold list must not be empty.")

    # Timeframe conversion factors to 1-minute equivalents
    timeframe_to_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }

    # Timeframe-specific cadence for ambiguity validation
    timeframe_to_cadence = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
    }

    out: dict[tuple[str, int, float], pd.DataFrame] = {}
    for timeframe in MODELED_TIMEFRAMES:
        bars_tf = bars_by_timeframe[timeframe]
        minutes_per_bar = timeframe_to_minutes[timeframe]

        for horizon in horizons:
            # Convert horizon_bars to 1-minute equivalents for ambiguity classification
            horizon_minutes_1m = int(horizon) * minutes_per_bar
            ambiguity_mask = classify_ambiguity(
                pd.DatetimeIndex(bars_tf.index),
                horizon_bars=horizon_minutes_1m,
                config=config,
                expected_cadence=timeframe_to_cadence[timeframe],
                validate_cadence=(timeframe != "1d"),  # Skip cadence validation for daily data due to DST transitions
            )

            for threshold in thresholds_to_use:
                labels = _labels_for_horizon_threshold(
                    bars_tf,
                    horizon_bars=int(horizon),
                    threshold=float(threshold),
                    ambiguous_mask=ambiguity_mask,
                )
                out[(timeframe, int(horizon), float(threshold))] = labels

    LOGGER.info(
        "generated_labels_multi_timeframe",
        extra={
            "event": "generated_labels_multi_timeframe",
            "horizon_mode": horizon_mode,
            "timeframes": list(MODELED_TIMEFRAMES),
            "horizons": list(horizons),
            "threshold_count": len(thresholds_to_use),
            "total_combinations": len(MODELED_TIMEFRAMES) * len(horizons) * len(thresholds_to_use),
        },
    )
    return out


def _standardize_1m_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = ("end_ts", "open", "high", "low", "close")
    missing = [col for col in required if col not in bars.columns]
    if missing:
        raise LabelingError(f"bars_1m missing required columns: {missing}")

    bars = bars.copy()
    bars["end_ts"] = pd.to_datetime(bars["end_ts"], errors="raise")
    end_ts = pd.DatetimeIndex(bars["end_ts"])
    if end_ts.tz is None:
        raise LabelingError("bars_1m end_ts must be timezone-aware.")
    if end_ts.has_duplicates:
        raise LabelingError("bars_1m must not contain duplicate end_ts values.")
    if not end_ts.is_monotonic_increasing:
        bars = bars.sort_values("end_ts")
        end_ts = pd.DatetimeIndex(bars["end_ts"])
        if not end_ts.is_monotonic_increasing:
            raise LabelingError("bars_1m end_ts must be monotonic increasing.")

    bars.set_index("end_ts", inplace=True, drop=False)
    return bars


def _labels_for_horizon_threshold(
    bars: pd.DataFrame,
    *,
    horizon_bars: int,
    threshold: float,
    ambiguous_mask: pd.Series,
) -> pd.DataFrame:
    if horizon_bars <= 0:
        raise LabelingError("horizon_bars must be positive.")
    if threshold <= 0.0:
        raise LabelingError("threshold must be positive.")

    end_ts = pd.DatetimeIndex(bars.index)
    close = bars["close"].to_numpy(dtype=np.float64)
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)

    n = len(bars)
    event_flag = np.zeros(n, dtype=np.int64)
    event_start_offset = np.full(n, -1, dtype=np.int64)
    maturity_offset = np.full(n, -1, dtype=np.int64)
    future_low = np.full(n, np.nan, dtype=np.float64)
    future_high = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if bool(ambiguous_mask.iloc[i]):
            continue

        ref_close = close[i]
        window_start = i + 1
        window_end = i + horizon_bars
        if window_end >= n:
            raise LabelingError("Ambiguity mask allowed an out-of-range horizon window.")

        slice_high = high[window_start : window_end + 1]
        slice_low = low[window_start : window_end + 1]

        f_high = float(np.max(slice_high))
        f_low = float(np.min(slice_low))
        future_high[i] = f_high
        future_low[i] = f_low

        up_excursion = abs(f_high - ref_close)
        down_excursion = abs(ref_close - f_low)
        max_excursion = max(up_excursion, down_excursion)

        if max_excursion >= threshold:
            event_flag[i] = 1

        start = _find_event_start_offset(
            ref_close=ref_close,
            highs=slice_high,
            lows=slice_low,
            threshold=threshold,
        )
        maturity = _find_maturity_offset(
            ref_close=ref_close,
            highs=slice_high,
            lows=slice_low,
        )

        event_start_offset[i] = start
        maturity_offset[i] = maturity

    df = pd.DataFrame(
        {
            "reference_ts": end_ts,
            "horizon_bars": np.full(n, int(horizon_bars), dtype=np.int64),
            "threshold": np.full(n, float(threshold), dtype=np.float64),
            "event_flag": event_flag,
            "event_start_offset": event_start_offset,
            "maturity_offset": maturity_offset,
            "future_low": future_low,
            "future_high": future_high,
            "ambiguous": ambiguous_mask.to_numpy(dtype=bool),
        }
    )
    return df


def _find_event_start_offset(
    *,
    ref_close: float,
    highs: np.ndarray,
    lows: np.ndarray,
    threshold: float,
) -> int:
    running_high = highs[0]
    running_low = lows[0]
    for k in range(1, len(highs) + 1):
        running_high = float(max(running_high, highs[k - 1]))
        running_low = float(min(running_low, lows[k - 1]))
        up_excursion = abs(running_high - ref_close)
        down_excursion = abs(ref_close - running_low)
        if max(up_excursion, down_excursion) >= threshold:
            return k
    return -1


def _find_maturity_offset(
    *,
    ref_close: float,
    highs: np.ndarray,
    lows: np.ndarray,
) -> int:
    running_high = highs[0]
    running_low = lows[0]
    best_k = 1
    best_excursion = -1.0
    for k in range(1, len(highs) + 1):
        running_high = float(max(running_high, highs[k - 1]))
        running_low = float(min(running_low, lows[k - 1]))
        up_excursion = abs(running_high - ref_close)
        down_excursion = abs(ref_close - running_low)
        excursion = max(up_excursion, down_excursion)
        if excursion > best_excursion:
            best_excursion = excursion
            best_k = k
    return best_k

