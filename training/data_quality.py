"""U2 data quality validation for `1m` OHLC sequences."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from training.calendar import normalize_calendar
from training.config_schema import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class DataQualityError(ValueError):
    """Raised when data quality validation fails."""


def validate_bar_sequence(frame_1m: pd.DataFrame, config: RuntimeConfig) -> pd.DataFrame:
    """Validate bar sequence rules for `1m` OHLC data.

    This enforces strict timestamp ordering, duplicates, minute alignment, and
    missing bar detection without silently correcting any inputs.

    Args:
        frame_1m: `1m` bars indexed by `end_ts`.
        config: Validated runtime configuration.

    Returns:
        The input frame (unchanged) when validation passes.

    Raises:
        DataQualityError: If any strict sequence validation fails.
    """
    if "end_ts" not in frame_1m.columns:
        raise DataQualityError("Expected `end_ts` column in 1m frame.")
    if len(frame_1m) == 0:
        raise DataQualityError("1m OHLC frame must not be empty.")

    end_ts = pd.DatetimeIndex(frame_1m["end_ts"])
    if end_ts.tz is None:
        raise DataQualityError("end_ts must be timezone-aware.")

    if not end_ts.is_monotonic_increasing:
        raise DataQualityError("1m bar timestamps must be strictly increasing.")
    if end_ts.has_duplicates:
        duplicate = end_ts[end_ts.duplicated()][0]
        raise DataQualityError(f"Duplicate timestamp detected: {duplicate.isoformat()}")

    if (end_ts.second != 0).any() or (end_ts.microsecond != 0).any():
        raise DataQualityError("All OHLC timestamps must be aligned to exact 1-minute boundaries.")

    normalize_calendar(frame_1m, config)

    expected = _build_expected_minutes(end_ts.min(), end_ts.max(), config)
    missing = expected.difference(end_ts)
    if len(missing) > 0:
        first_missing = missing.sort_values()[0]
        raise DataQualityError(
            "Missing 1m bars detected within expected session minutes. "
            f"missing_count={len(missing)} first_missing={first_missing.isoformat()}"
        )

    LOGGER.info(
        "validated_bar_sequence",
        extra={
            "event": "validated_bar_sequence",
            "row_count": int(len(frame_1m)),
            "start_ts": end_ts.min().isoformat(),
            "end_ts": end_ts.max().isoformat(),
        },
    )
    return frame_1m


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start_minute: int
    end_minute: int


def _build_expected_minutes(start_ts: pd.Timestamp, end_ts: pd.Timestamp, config: RuntimeConfig) -> pd.DatetimeIndex:
    runtime_tz = _require_timezone(config.time.runtime_timezone, field_name="time.runtime_timezone")
    if start_ts.tz is None:
        raise DataQualityError("start_ts must be timezone-aware.")

    start_ts = start_ts.tz_convert(runtime_tz)
    end_ts = end_ts.tz_convert(runtime_tz)

    all_minutes = pd.date_range(start=start_ts, end=end_ts, freq="min", tz=runtime_tz)

    if config.sessions.weekend_policy == "exclude":
        all_minutes = all_minutes[all_minutes.weekday < 5]

    session_windows = tuple(
        SessionWindow(
            name=definition.name,
            start_minute=_parse_hhmm(definition.start),
            end_minute=_parse_hhmm(definition.end),
        )
        for definition in config.sessions.definitions
    )

    minute_of_day = all_minutes.hour * 60 + all_minutes.minute
    session_mask = pd.Series(False, index=range(len(all_minutes)))
    for window in session_windows:
        if window.start_minute <= window.end_minute:
            window_mask = (minute_of_day >= window.start_minute) & (minute_of_day <= window.end_minute)
        else:
            window_mask = (minute_of_day >= window.start_minute) | (minute_of_day <= window.end_minute)
        session_mask = session_mask | pd.Series(window_mask)

    return all_minutes[session_mask.to_numpy()]


def _parse_hhmm(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 2:
        raise DataQualityError("Session times must use HH:MM format.")
    hour_str, minute_str = parts
    try:
        hour = int(hour_str)
        minute = int(minute_str)
    except ValueError as exc:
        raise DataQualityError("Session times must use numeric HH:MM values.") from exc
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise DataQualityError("Session times must use a valid 24-hour HH:MM time.")
    return hour * 60 + minute


def _require_timezone(value: str, *, field_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise DataQualityError(f"Unresolved timezone in {field_name}: {value}") from exc
