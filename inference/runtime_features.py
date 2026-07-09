"""U10 runtime feature assembly with training-parity validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final
from zoneinfo import ZoneInfo

import pandas as pd
import torch

from models.common import TIMEFRAMES
from training.config_schema import RuntimeConfig
from training.contracts import PredictionRequestContract
from training.data_quality import validate_bar_sequence
from training.features import build_features
from training.resample import resample_timeframes
from training.windowing import build_windows

EXPECTED_TIMEFRAMES: Final[set[str]] = set(TIMEFRAMES)


class RuntimeFeatureError(ValueError):
    """Raised when runtime feature assembly fails."""


@dataclass(frozen=True)
class RuntimeWindow:
    """Typed runtime window bundle ready for model inference."""

    reference_ts: datetime
    reference_close: torch.Tensor
    windows_by_timeframe: dict[str, torch.Tensor]
    feature_names_by_timeframe: dict[str, tuple[str, ...]]


def build_runtime_window(
    request: PredictionRequestContract,
    config: RuntimeConfig,
    lookbacks_by_timeframe: dict[str, int],
    *,
    current_time: datetime | None = None,
) -> RuntimeWindow:
    """Build runtime tensors with the same deterministic path used in training.

    Args:
        request: Validated prediction request.
        config: Validated runtime configuration.
        lookbacks_by_timeframe: Trailing lookbacks for each modeled timeframe.
        current_time: Optional current time override for closed-bar validation.

    Returns:
        Runtime window bundle for the latest fully closed bar.
    """
    _validate_lookbacks(lookbacks_by_timeframe)
    if request.instrument_id != config.instrument.instrument_id:
        raise RuntimeFeatureError(
            "request.instrument_id must match config.instrument.instrument_id for inference."
        )

    frame_1m = _request_to_frame(request)
    validate_bar_sequence(frame_1m, config)
    _validate_latest_bar_closed(frame_1m, config=config, current_time=current_time)

    bars_by_timeframe = resample_timeframes(frame_1m, config)
    features_by_timeframe = build_features(bars_by_timeframe, config)
    windows = build_windows(features_by_timeframe, lookbacks_by_timeframe=lookbacks_by_timeframe)

    reference_ts = frame_1m["end_ts"].iloc[-1].to_pydatetime()
    reference_close = torch.tensor([float(frame_1m["close"].iloc[-1])], dtype=torch.float32)
    windows_by_timeframe = {
        timeframe: torch.tensor(feature_windows.windows[-1: ], dtype=torch.float32)
        for timeframe, feature_windows in windows.items()
    }
    feature_names_by_timeframe = {
        timeframe: feature_windows.feature_names for timeframe, feature_windows in windows.items()
    }
    return RuntimeWindow(
        reference_ts=reference_ts,
        reference_close=reference_close,
        windows_by_timeframe=windows_by_timeframe,
        feature_names_by_timeframe=feature_names_by_timeframe,
    )


def _request_to_frame(request: PredictionRequestContract) -> pd.DataFrame:
    rows = [
        {
            "end_ts": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
        }
        for bar in request.bars_1m
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeFeatureError("request.bars_1m must not be empty.")
    frame["end_ts"] = pd.to_datetime(frame["end_ts"], errors="raise")
    return frame


def _validate_lookbacks(lookbacks_by_timeframe: dict[str, int]) -> None:
    if set(lookbacks_by_timeframe.keys()) != EXPECTED_TIMEFRAMES:
        raise RuntimeFeatureError(f"lookbacks_by_timeframe must contain exactly {TIMEFRAMES}.")
    for timeframe, lookback in lookbacks_by_timeframe.items():
        if lookback <= 0:
            raise RuntimeFeatureError(f"lookback must be positive for timeframe={timeframe}.")


def _validate_latest_bar_closed(
    frame_1m: pd.DataFrame,
    *,
    config: RuntimeConfig,
    current_time: datetime | None,
) -> None:
    latest_end_ts = pd.Timestamp(frame_1m["end_ts"].iloc[-1])
    runtime_tz = ZoneInfo(config.time.runtime_timezone)
    latest_runtime = latest_end_ts.tz_convert(runtime_tz)

    effective_now = current_time if current_time is not None else datetime.now(timezone.utc)
    if effective_now.tzinfo is None or effective_now.utcoffset() is None:
        raise RuntimeFeatureError("current_time must be timezone-aware when provided.")
    now_runtime = effective_now.astimezone(runtime_tz)
    latest_allowed = now_runtime.replace(second=0, microsecond=0)
    if latest_runtime > latest_allowed:
        raise RuntimeFeatureError("inference must reject incomplete recent bars.")
