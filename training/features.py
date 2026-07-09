"""Deterministic feature engineering (U3)."""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

from training.config_schema import RuntimeConfig
from training.feature_registry import list_enabled_features, resolve_feature

LOGGER = logging.getLogger(__name__)

REQUIRED_BAR_COLUMNS: Final[tuple[str, ...]] = ("end_ts", "open", "high", "low", "close")


class FeatureEngineeringError(ValueError):
    """Raised when feature engineering fails."""


def build_features(
    bars_by_timeframe: dict[str, pd.DataFrame],
    config: RuntimeConfig,
) -> dict[str, pd.DataFrame]:
    """Build deterministic features for every modeled timeframe.

    Args:
        bars_by_timeframe: Output of U2 resampling keyed by timeframe.
        config: Validated runtime configuration.

    Returns:
        Mapping timeframe -> feature frame indexed by `end_ts`.

    Raises:
        FeatureEngineeringError: If inputs are invalid, features are unknown,
            or any computation is non-causal or inconsistent.
    """
    expected = {"1m", "5m", "15m", "1h", "4h", "1d"}
    if set(bars_by_timeframe.keys()) != expected:
        raise FeatureEngineeringError(
            f"bars_by_timeframe must contain exactly the modeled stack {sorted(expected)}."
        )

    enabled_raw = tuple(config.features.enabled_features)
    deterministic_raw = tuple(config.features.deterministic_derived_features)
    missing = [name for name in deterministic_raw if name not in enabled_raw]
    if missing:
        raise FeatureEngineeringError(
            "deterministic_derived_features must be a subset of enabled_features. "
            f"Missing from enabled_features: {missing}"
        )

    enabled = list_enabled_features(enabled_raw)

    out: dict[str, pd.DataFrame] = {}
    for timeframe, bars in bars_by_timeframe.items():
        out[timeframe] = _build_timeframe_features(bars, config=config, timeframe=timeframe, enabled=enabled)

    LOGGER.info(
        "built_features",
        extra={
            "event": "built_features",
            "timeframes": sorted(out.keys()),
            "enabled_features_count": len(enabled),
        },
    )
    return out


def _build_timeframe_features(
    bars: pd.DataFrame,
    *,
    config: RuntimeConfig,
    timeframe: str,
    enabled: tuple[str, ...],
) -> pd.DataFrame:
    bars = _standardize_bar_frame(bars)

    base = _compute_base_features(bars)
    calendar = _compute_calendar_features(bars)
    sessions = _compute_session_flags(bars, config=config)

    feature_pool = pd.concat([base, calendar, sessions], axis=1)
    ordered_columns = []
    for name in enabled:
        resolved = resolve_feature(name)
        if resolved.rolling_op is None:
            if name not in feature_pool.columns:
                raise FeatureEngineeringError(f"Unknown enabled feature '{name}' for timeframe={timeframe}.")
            ordered_columns.append(name)
            continue

        if resolved.base_feature is None or resolved.window is None:
            raise FeatureEngineeringError(f"Malformed rolling feature name: {name}")
        if resolved.base_feature not in feature_pool.columns:
            raise FeatureEngineeringError(
                f"Rolling feature '{name}' references unknown base feature '{resolved.base_feature}'."
            )
        rolling_series = _compute_causal_rolling(
            feature_pool[resolved.base_feature],
            op=resolved.rolling_op,
            window=resolved.window,
        )
        feature_pool[name] = rolling_series
        ordered_columns.append(name)

    frame = feature_pool[ordered_columns].copy()
    frame.index = bars.index
    frame.index.name = "end_ts"
    if frame.isna().any().any():
        missing_cols = frame.columns[frame.isna().any()].tolist()
        raise FeatureEngineeringError(f"Feature computation produced NaNs for columns: {missing_cols}")
    return frame


def _standardize_bar_frame(bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame):
        raise FeatureEngineeringError("bars must be a pandas DataFrame.")
    missing = [col for col in REQUIRED_BAR_COLUMNS if col not in bars.columns]
    if missing:
        raise FeatureEngineeringError(f"bars missing required columns: {missing}")

    bars = bars.copy()
    bars["end_ts"] = pd.to_datetime(bars["end_ts"], errors="raise")
    index = pd.DatetimeIndex(bars["end_ts"])
    if index.tz is None:
        raise FeatureEngineeringError("end_ts must be timezone-aware.")
    if index.has_duplicates:
        raise FeatureEngineeringError("bars must not contain duplicate end_ts values.")
    if not index.is_monotonic_increasing:
        bars = bars.sort_values("end_ts")
        index = pd.DatetimeIndex(bars["end_ts"])
        if not index.is_monotonic_increasing:
            raise FeatureEngineeringError("bars end_ts must be monotonic increasing.")

    bars.set_index("end_ts", inplace=True, drop=False)
    return bars


def _compute_base_features(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"].astype("float64")
    open_ = bars["open"].astype("float64")
    high = bars["high"].astype("float64")
    low = bars["low"].astype("float64")

    prev_close = close.shift(1)
    if prev_close.isna().iloc[0] is False:
        raise FeatureEngineeringError("Unexpected non-NaN prev_close at first row.")

    returns = (close / prev_close) - 1.0
    returns.iloc[0] = 0.0

    log_return = np.log(close) - np.log(prev_close)
    log_return.iloc[0] = 0.0

    hl_range = high - low
    oc_change = close - open_

    body = (close - open_).abs()
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    body_arr = body.to_numpy()
    upper_arr = upper_wick.to_numpy()
    lower_arr = lower_wick.to_numpy()
    upper_wick_body_ratio = np.divide(
        upper_arr,
        body_arr,
        out=np.zeros_like(upper_arr, dtype=np.float64),
        where=body_arr != 0.0,
    )
    lower_wick_body_ratio = np.divide(
        lower_arr,
        body_arr,
        out=np.zeros_like(lower_arr, dtype=np.float64),
        where=body_arr != 0.0,
    )

    return pd.DataFrame(
        {
            "return": returns.to_numpy(),
            "log_return": log_return.to_numpy(),
            "hl_range": hl_range.to_numpy(),
            "oc_change": oc_change.to_numpy(),
            "body": body.to_numpy(),
            "upper_wick": upper_wick.to_numpy(),
            "lower_wick": lower_wick.to_numpy(),
            "upper_wick_body_ratio": upper_wick_body_ratio,
            "lower_wick_body_ratio": lower_wick_body_ratio,
        },
        index=bars.index,
    )


def _compute_calendar_features(bars: pd.DataFrame) -> pd.DataFrame:
    end_ts = pd.DatetimeIndex(bars.index)
    return pd.DataFrame(
        {
            "dow": end_ts.dayofweek.astype("int64"),
            "hour": end_ts.hour.astype("int64"),
            "minute": end_ts.minute.astype("int64"),
            "dom": end_ts.day.astype("int64"),
            "month": end_ts.month.astype("int64"),
        },
        index=bars.index,
    )


def _compute_session_flags(bars: pd.DataFrame, *, config: RuntimeConfig) -> pd.DataFrame:
    end_ts = pd.DatetimeIndex(bars.index).tz_convert(config.time.runtime_timezone)
    minute_of_day = end_ts.hour * 60 + end_ts.minute

    flags: dict[str, np.ndarray] = {}
    for definition in config.sessions.definitions:
        start_minute = _parse_hhmm(definition.start)
        end_minute = _parse_hhmm(definition.end)
        if start_minute <= end_minute:
            mask = (minute_of_day >= start_minute) & (minute_of_day <= end_minute)
        else:
            mask = (minute_of_day >= start_minute) | (minute_of_day <= end_minute)
        flags[f"session_{definition.name}"] = mask.astype("int64")

    return pd.DataFrame(flags, index=bars.index)


def _compute_causal_rolling(series: pd.Series, *, op: str, window: int) -> pd.Series:
    if window <= 0:
        raise FeatureEngineeringError("window must be positive.")
    rolling = series.rolling(window=window, min_periods=window, center=False)
    if op == "mean":
        out = rolling.mean()
    elif op == "std":
        out = rolling.std(ddof=0)
    elif op == "min":
        out = rolling.min()
    elif op == "max":
        out = rolling.max()
    else:
        raise FeatureEngineeringError(f"Unsupported rolling op: {op}")

    out = out.copy()
    out.iloc[: window - 1] = out.iloc[: window - 1].fillna(0.0)
    if out.isna().any():
        raise FeatureEngineeringError("Rolling computation produced NaNs after deterministic fill.")
    return out


def _parse_hhmm(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 2:
        raise FeatureEngineeringError("Session times must use HH:MM format.")
    hour_str, minute_str = parts
    try:
        hour = int(hour_str)
        minute = int(minute_str)
    except ValueError as exc:
        raise FeatureEngineeringError("Session times must use numeric HH:MM values.") from exc
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise FeatureEngineeringError("Session times must use a valid 24-hour HH:MM time.")
    return hour * 60 + minute
