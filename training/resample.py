"""U2 deterministic resampling for Sebuleni Pro Max AI."""

from __future__ import annotations

import logging
from datetime import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from training.config_schema import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class ResampleError(ValueError):
    """Raised when deterministic resampling fails."""


def resample_timeframes(frame_1m: pd.DataFrame, config: RuntimeConfig) -> dict[str, pd.DataFrame]:
    """Resample validated `1m` bars into the active multi-timeframe stack.

    The returned mapping contains:
    - `1m`: validated `1m` bars as a standardized bar frame
    - `5m`, `15m`, `1h`, `4h`, `1d`: deterministically derived OHLC bars

    Args:
        frame_1m: Validated `1m` OHLC frame.
        config: Validated runtime configuration.

    Returns:
        Mapping of timeframe string -> standardized OHLC bar frame with columns:
        `timeframe`, `start_ts`, `end_ts`, `open`, `high`, `low`, `close`.

    Raises:
        ResampleError: If required columns are missing or resampling fails.
    """
    required = {"end_ts", "open", "high", "low", "close"}
    if not required.issubset(frame_1m.columns):
        missing = sorted(required - set(frame_1m.columns))
        raise ResampleError(f"1m frame missing required columns: {missing}")
    if len(frame_1m) == 0:
        raise ResampleError("1m frame must not be empty.")

    frame_1m = frame_1m.copy()
    if frame_1m.index.name != "end_ts":
        frame_1m.set_index("end_ts", inplace=True, drop=False)
    frame_1m = frame_1m.sort_index()

    out: dict[str, pd.DataFrame] = {}
    out["1m"] = _standardize_1m(frame_1m)

    for timeframe in config.resampling.target_timeframes:
        if timeframe == "1d":
            out["1d"] = _resample_daily_close(frame_1m, config)
        else:
            out[timeframe] = _resample_fixed_interval(frame_1m, timeframe)

    LOGGER.info(
        "resampled_timeframes",
        extra={
            "event": "resampled_timeframes",
            "timeframes": ["1m", *list(config.resampling.target_timeframes)],
            "rows_1m": int(len(out["1m"])),
            "rows_5m": int(len(out.get("5m", []))),
            "rows_15m": int(len(out.get("15m", []))),
            "rows_1h": int(len(out.get("1h", []))),
            "rows_4h": int(len(out.get("4h", []))),
            "rows_1d": int(len(out.get("1d", []))),
        },
    )
    return out


def _standardize_1m(frame_1m: pd.DataFrame) -> pd.DataFrame:
    frame = frame_1m[["end_ts", "open", "high", "low", "close"]].copy()
    frame["timeframe"] = "1m"
    frame["start_ts"] = frame["end_ts"] - pd.Timedelta(minutes=1)
    result = frame[["timeframe", "start_ts", "end_ts", "open", "high", "low", "close"]].reset_index(drop=True)
    result.set_index("end_ts", inplace=True, drop=False)
    return result


def _resample_fixed_interval(frame_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = _timeframe_to_pandas_rule(timeframe)
    period = _timeframe_to_timedelta(timeframe)

    segments = _split_contiguous_segments(frame_1m.index)
    frames: list[pd.DataFrame] = []
    for segment_start, segment_end in segments:
        segment = frame_1m.loc[segment_start:segment_end]
        ohlc = (
            segment[["open", "high", "low", "close"]]
            .resample(rule, label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        )

        if ohlc.isna().any().any():
            first_bad = ohlc[ohlc.isna().any(axis=1)].index[0]
            raise ResampleError(f"Empty resample bucket detected for timeframe={timeframe} at {first_bad.isoformat()}")

        ohlc = ohlc.reset_index().rename(columns={"index": "end_ts"})
        ohlc["timeframe"] = timeframe
        ohlc["start_ts"] = ohlc["end_ts"] - period
        frames.append(ohlc[["timeframe", "start_ts", "end_ts", "open", "high", "low", "close"]])

    if not frames:
        raise ResampleError(f"No data available to resample timeframe={timeframe}")

    result = pd.concat(frames, ignore_index=True).sort_values("end_ts").reset_index(drop=True)
    result.set_index("end_ts", inplace=True, drop=False)
    return result


def _resample_daily_close(frame_1m: pd.DataFrame, config: RuntimeConfig) -> pd.DataFrame:
    close_tz = _require_timezone(config.time.daily_close.timezone, field_name="time.daily_close.timezone")
    runtime_tz = _require_timezone(config.time.runtime_timezone, field_name="time.runtime_timezone")
    close_time = _parse_close_time(config.time.daily_close.time)

    end_ts_runtime = pd.DatetimeIndex(frame_1m.index).tz_convert(runtime_tz)
    end_ts_close = end_ts_runtime.tz_convert(close_tz)

    close_midnight = end_ts_close.normalize()
    close_dt = close_midnight + pd.Timedelta(hours=close_time.hour, minutes=close_time.minute)
    # Use DateOffset instead of Timedelta to handle DST transitions correctly
    bucket_end_close = close_dt.where(end_ts_close <= close_dt, close_dt + pd.DateOffset(days=1))
    bucket_end_runtime = bucket_end_close.tz_convert(runtime_tz)

    grouped = frame_1m.copy()
    grouped["bucket_end"] = bucket_end_runtime

    aggregated = grouped.groupby("bucket_end", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )

    aggregated = aggregated.reset_index().rename(columns={"bucket_end": "end_ts"})
    aggregated["timeframe"] = "1d"
    # Use DateOffset instead of Timedelta to handle DST transitions correctly
    aggregated["start_ts"] = aggregated["end_ts"].apply(lambda ts: (ts.tz_convert(close_tz) - pd.DateOffset(days=1)).tz_convert(runtime_tz))
    result = aggregated[["timeframe", "start_ts", "end_ts", "open", "high", "low", "close"]]
    result.set_index("end_ts", inplace=True, drop=False)
    return result


def _split_contiguous_segments(index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    diffs = index.to_series().diff()
    breaks = diffs != pd.Timedelta(minutes=1)
    segment_ids = breaks.cumsum()

    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, group in index.to_series().groupby(segment_ids):
        segments.append((group.iloc[0], group.iloc[-1]))
    return segments


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    if timeframe == "5m":
        return pd.Timedelta(minutes=5)
    if timeframe == "15m":
        return pd.Timedelta(minutes=15)
    if timeframe == "1h":
        return pd.Timedelta(hours=1)
    if timeframe == "4h":
        return pd.Timedelta(hours=4)
    raise ResampleError(f"Unsupported timeframe: {timeframe}")


def _timeframe_to_pandas_rule(timeframe: str) -> str:
    if timeframe == "5m":
        return "5min"
    if timeframe == "15m":
        return "15min"
    if timeframe == "1h":
        return "1h"
    if timeframe == "4h":
        return "4h"
    raise ResampleError(f"Unsupported timeframe: {timeframe}")


def _parse_close_time(value: str) -> time:
    parts = value.split(":")
    if len(parts) != 2:
        raise ResampleError("daily_close.time must use HH:MM format.")
    hour_str, minute_str = parts
    try:
        hour = int(hour_str)
        minute = int(minute_str)
    except ValueError as exc:
        raise ResampleError("daily_close.time must use numeric HH:MM values.") from exc
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ResampleError("daily_close.time must be a valid 24-hour time.")
    return time(hour=hour, minute=minute)


def _require_timezone(value: str, *, field_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise ResampleError(f"Unresolved timezone in {field_name}: {value}") from exc
