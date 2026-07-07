"""U2 calendar normalization and session validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from training.config_schema import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class CalendarError(ValueError):
    """Raised when calendar normalization fails."""


def normalize_calendar(frame_1m: pd.DataFrame, config: RuntimeConfig) -> pd.DataFrame:
    """Validate calendar and session constraints for a `1m` OHLC frame.

    No filtering is performed. Bars that violate constraints result in a
    validation exception to avoid silent corrections.

    Args:
        frame_1m: `1m` bars indexed by `end_ts` in runtime timezone.
        config: Validated runtime configuration.

    Returns:
        The input frame (unchanged) when all constraints are satisfied.

    Raises:
        CalendarError: If session/weekend/holiday rules fail.
    """
    if "end_ts" not in frame_1m.columns:
        raise CalendarError("Expected `end_ts` column in 1m frame.")

    runtime_tz = _require_timezone(config.time.runtime_timezone, field_name="time.runtime_timezone")
    end_ts = pd.DatetimeIndex(frame_1m["end_ts"])
    if end_ts.tz is None:
        raise CalendarError("end_ts must be timezone-aware.")

    end_ts = end_ts.tz_convert(runtime_tz)

    if config.sessions.weekend_policy == "exclude":
        weekend_mask = end_ts.weekday >= 5
        if weekend_mask.any():
            offending = end_ts[weekend_mask][0]
            raise CalendarError(f"Weekend bars are not permitted by weekend_policy=exclude: {offending.isoformat()}")

    if config.sessions.holiday_policy == "exclude":
        raise CalendarError(
            "holiday_policy=exclude is configured but no deterministic holiday calendar source is provided "
            "by the approved U1 contracts. Set holiday_policy=include for U2, or provide an approved holiday "
            "calendar mechanism in a future SRS amendment."
        )

    session_windows = _build_session_windows(config)
    session_mask = _compute_session_mask(end_ts, session_windows)
    if not session_mask.all():
        offending = end_ts[~session_mask][0]
        raise CalendarError(f"Bar timestamp falls outside configured session windows: {offending.isoformat()}")

    LOGGER.info(
        "normalized_calendar",
        extra={
            "event": "normalized_calendar",
            "calendar_name": config.sessions.calendar_name,
            "weekend_policy": config.sessions.weekend_policy,
            "holiday_policy": config.sessions.holiday_policy,
            "session_definitions": len(config.sessions.definitions),
        },
    )
    return frame_1m


@dataclass(frozen=True)
class SessionWindow:
    """Pre-parsed session window."""

    name: str
    start_minute: int
    end_minute: int
    start_day: int | None = None
    end_day: int | None = None


def _build_session_windows(config: RuntimeConfig) -> tuple[SessionWindow, ...]:
    windows: list[SessionWindow] = []
    for definition in config.sessions.definitions:
        start_minute = _parse_hhmm(definition.start, field_name=f"sessions.definitions[{definition.name}].start")
        end_minute = _parse_hhmm(definition.end, field_name=f"sessions.definitions[{definition.name}].end")
        windows.append(SessionWindow(
            name=definition.name,
            start_minute=start_minute,
            end_minute=end_minute,
            start_day=definition.start_day,
            end_day=definition.end_day,
        ))
    return tuple(windows)


def _compute_session_mask(timestamps: pd.DatetimeIndex, windows: tuple[SessionWindow, ...]) -> pd.Series:
    minute_of_day = timestamps.hour * 60 + timestamps.minute
    day_of_week = timestamps.weekday
    mask = pd.Series(False, index=range(len(timestamps)))
    for window in windows:
        # Time-based mask
        if window.start_minute <= window.end_minute:
            time_mask = (minute_of_day >= window.start_minute) & (minute_of_day <= window.end_minute)
        else:
            time_mask = (minute_of_day >= window.start_minute) | (minute_of_day <= window.end_minute)

        # Day-of-week mask (if specified)
        if window.start_day is not None and window.end_day is not None:
            if window.start_day <= window.end_day:
                # Normal range (e.g., Monday=0 to Friday=4)
                day_mask = (day_of_week >= window.start_day) & (day_of_week <= window.end_day)
            else:
                # Cross-weekend boundary (e.g., Sunday=6 to Friday=4)
                day_mask = (day_of_week >= window.start_day) | (day_of_week <= window.end_day)
            window_mask = time_mask & day_mask
        else:
            # No day constraints, just time
            window_mask = time_mask

        mask = mask | pd.Series(window_mask)
    return mask.astype(bool)


def _parse_hhmm(value: str, *, field_name: str) -> int:
    parts = value.split(":")
    if len(parts) != 2:
        raise CalendarError(f"{field_name} must use HH:MM format.")
    hour_str, minute_str = parts
    try:
        hour = int(hour_str)
        minute = int(minute_str)
    except ValueError as exc:
        raise CalendarError(f"{field_name} must use numeric HH:MM.") from exc
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise CalendarError(f"{field_name} must use a valid 24-hour HH:MM time.")
    return hour * 60 + minute


def _require_timezone(value: str, *, field_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise CalendarError(f"Unresolved timezone in {field_name}: {value}") from exc

