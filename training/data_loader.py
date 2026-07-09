"""U2 OHLC ingestion for Sebuleni Pro Max AI."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from training.config_schema import RuntimeConfig

LOGGER = logging.getLogger(__name__)

REQUIRED_OHLC_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close")


class OhlcLoadError(ValueError):
    """Raised when OHLC ingestion fails."""


def load_ohlc_frame(config: RuntimeConfig) -> pd.DataFrame:
    """Load and validate `1m` OHLC bars for the configured instrument.

    This function performs strict schema validation and deterministic timezone
    normalization. It does not perform session/weekend/holiday validation or
    resampling; those responsibilities belong to `normalize_calendar()` and
    `resample_timeframes()`.

    The returned frame uses `end_ts` as the timestamp column, representing the
    fully closed bar end timestamp. The index is set to `end_ts`.

    Args:
        config: Validated runtime configuration.

    Returns:
        A DataFrame with columns: `end_ts`, `open`, `high`, `low`, `close`.

    Raises:
        OhlcLoadError: If the file is missing, malformed, unsupported, or fails
            strict schema and timezone validation.
    """
    path = Path(config.data_source.ohlc_path)
    if not path.exists():
        raise OhlcLoadError(f"OHLC file does not exist: {path}")
    if not path.is_file():
        raise OhlcLoadError(f"OHLC path must point to a file: {path}")

    source_timezone = _require_timezone(config.time.source_timezone, field_name="time.source_timezone")
    runtime_timezone = _require_timezone(config.time.runtime_timezone, field_name="time.runtime_timezone")

    frame = _read_ohlc_file(path)
    _validate_schema(frame, required_columns=REQUIRED_OHLC_COLUMNS)

    frame = frame[list(REQUIRED_OHLC_COLUMNS)].copy()
    frame["timestamp"] = _parse_timestamps(frame["timestamp"], source_timezone=source_timezone)
    frame["timestamp"] = frame["timestamp"].dt.tz_convert(runtime_timezone)
    frame.rename(columns={"timestamp": "end_ts"}, inplace=True)

    for column in ("open", "high", "low", "close"):
        frame[column] = _coerce_numeric(frame[column], column_name=column)

    _validate_price_consistency(frame)

    frame = frame.sort_values("end_ts").reset_index(drop=True)
    frame.set_index("end_ts", inplace=True, drop=False)

    LOGGER.info(
        "loaded_ohlc_frame",
        extra={
            "event": "loaded_ohlc_frame",
            "instrument_id": config.instrument.instrument_id,
            "ohlc_path": str(path),
            "row_count": int(len(frame)),
            "runtime_timezone": config.time.runtime_timezone,
        },
    )
    return frame


def _read_ohlc_file(path: Path) -> pd.DataFrame:
    """Read an OHLC file deterministically.

    Supported formats:
    - CSV (`.csv`)
    - Parquet (`.parquet`) only when a parquet engine is installed

    Args:
        path: Path to the OHLC file.

    Returns:
        Raw DataFrame.

    Raises:
        OhlcLoadError: If reading fails or format is unsupported.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            raise OhlcLoadError(f"Failed to read OHLC CSV: {path}") from exc
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise OhlcLoadError(
                "Parquet support requires an installed engine (pyarrow or fastparquet). "
                f"Unable to read: {path}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise OhlcLoadError(f"Failed to read OHLC parquet: {path}") from exc

    raise OhlcLoadError(f"Unsupported OHLC file format: {path.suffix}")


def _validate_schema(frame: pd.DataFrame, *, required_columns: Sequence[str]) -> None:
    """Validate strict OHLC schema.

    Args:
        frame: Input DataFrame.
        required_columns: Required column names.

    Raises:
        OhlcLoadError: If schema is missing/extra columns or not a DataFrame.
    """
    if not isinstance(frame, pd.DataFrame):
        raise OhlcLoadError("OHLC payload must be a tabular frame.")

    columns = tuple(str(col) for col in frame.columns)
    required_set = set(required_columns)
    column_set = set(columns)
    if column_set != required_set:
        missing = sorted(required_set - column_set)
        extra = sorted(column_set - required_set)
        raise OhlcLoadError(
            "OHLC schema mismatch. "
            f"Missing columns: {missing or 'none'}. Extra columns: {extra or 'none'}. "
            f"Required columns: {list(required_columns)}."
        )


def _require_timezone(value: str, *, field_name: str) -> ZoneInfo:
    """Validate and return a timezone instance.

    Args:
        value: IANA timezone string.
        field_name: Config field name for diagnostics.

    Returns:
        A ZoneInfo instance.

    Raises:
        OhlcLoadError: If the timezone cannot be resolved.
    """
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise OhlcLoadError(f"Unresolved timezone in {field_name}: {value}") from exc


def _parse_timestamps(series: pd.Series, *, source_timezone: ZoneInfo) -> pd.Series:
    """Parse timestamps into timezone-aware datetimes.

    Args:
        series: Timestamp column.
        source_timezone: Source timezone to localize naive timestamps.

    Returns:
        Timezone-aware pandas Series.

    Raises:
        OhlcLoadError: If parsing fails, timestamps are not 1-minute aligned, or
            DST ambiguity exists in the source timezone.
    """
    parsed = pd.to_datetime(series, errors="raise", utc=False)

    if getattr(parsed.dt, "tz", None) is None:
        try:
            parsed = parsed.dt.tz_localize(source_timezone, ambiguous="raise", nonexistent="raise")
        except Exception as exc:  # noqa: BLE001
            raise OhlcLoadError("Failed to localize naive timestamps to source timezone.") from exc

    if (parsed.dt.second != 0).any() or (parsed.dt.microsecond != 0).any():
        raise OhlcLoadError("All OHLC timestamps must be aligned to exact 1-minute boundaries.")

    return parsed


def _coerce_numeric(series: pd.Series, *, column_name: str) -> pd.Series:
    """Coerce a series to numeric float values.

    Args:
        series: Candidate numeric series.
        column_name: Column name for diagnostics.

    Returns:
        A float series.

    Raises:
        OhlcLoadError: If conversion fails or non-finite values exist.
    """
    numeric = pd.to_numeric(series, errors="raise").astype("float64")
    if numeric.isna().any():
        raise OhlcLoadError(f"Column '{column_name}' contains NaN/null values.")
    if (~np.isfinite(numeric.to_numpy())).any():
        raise OhlcLoadError(f"Column '{column_name}' contains non-finite values.")
    return numeric


def _validate_price_consistency(frame: pd.DataFrame) -> None:
    """Validate OHLC price relationships.

    Args:
        frame: OHLC DataFrame.

    Raises:
        OhlcLoadError: If any bar is inconsistent.
    """
    high = frame["high"]
    low = frame["low"]
    open_ = frame["open"]
    close = frame["close"]

    invalid_high = high < open_.where(open_ >= close, close)
    invalid_low = low > open_.where(open_ <= close, close)
    invalid_range = low > high

    if invalid_range.any():
        idx = int(invalid_range.idxmax())
        row = frame.loc[idx].to_dict()
        raise OhlcLoadError(f"Malformed bar: low > high at row {idx}: {row}")
    if invalid_high.any():
        idx = int(invalid_high.idxmax())
        row = frame.loc[idx].to_dict()
        raise OhlcLoadError(f"Malformed bar: high < max(open, close) at row {idx}: {row}")
    if invalid_low.any():
        idx = int(invalid_low.idxmax())
        row = frame.loc[idx].to_dict()
        raise OhlcLoadError(f"Malformed bar: low > min(open, close) at row {idx}: {row}")
