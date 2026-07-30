"""Shared runtime contracts for Sebuleni Pro Max AI."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[StrictStr, Field(min_length=1)]
Probability = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[StrictFloat, Field(gt=0.0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]


class ContractModel(BaseModel):
    """Base contract model with immutable validated state."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class OhlcBarContract(ContractModel):
    """Shared OHLC bar contract for public inputs."""

    timestamp: datetime
    open: StrictFloat
    high: StrictFloat
    low: StrictFloat
    close: StrictFloat

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        """Require timezone-aware timestamps.

        Args:
            value: Candidate timestamp.

        Returns:
            The validated timestamp.

        Raises:
            ValueError: If the timestamp is naive.
        """
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("bar timestamps must be timezone-aware.")
        return value

    @field_validator("open", "high", "low", "close")
    @classmethod
    def validate_finite_price(cls, value: float) -> float:
        """Require finite price values.

        Args:
            value: Candidate price.

        Returns:
            The validated price.

        Raises:
            ValueError: If the value is NaN or infinite.
        """
        if not math.isfinite(value):
            raise ValueError("OHLC values must be finite numbers.")
        return value


class PredictionRequestContract(ContractModel):
    """Shared prediction request contract."""

    instrument_id: NonEmptyStr
    bars_1m: Annotated[tuple[OhlcBarContract, ...], Field(min_length=1)]
    horizon_mode: Literal["single", "multi"]
    horizon_bars: PositiveInt | None = None
    threshold: PositiveFloat
    top_k_analogs: PositiveInt

    # New fields for unified heads
    requested_timeframes: tuple[Literal["1m", "5m", "15m", "1h", "4h"], ...] | None = None
    requested_horizons: tuple[PositiveInt, ...] | None = None

    @model_validator(mode="after")
    def validate_horizon_fields(self) -> "PredictionRequestContract":
        """Enforce horizon requirements from the approved request contract.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the horizon fields do not match the horizon mode.
        """
        if self.horizon_mode == "single" and self.horizon_bars is None:
            raise ValueError("horizon_bars is required when horizon_mode is 'single'.")
        if self.horizon_mode == "multi" and self.horizon_bars is not None:
            raise ValueError("horizon_bars must not be provided when horizon_mode is 'multi'.")
        return self

    @model_validator(mode="after")
    def validate_unified_fields(self) -> "PredictionRequestContract":
        """Validate that requested timeframes/horizons are configured.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the requested timeframes or horizons are invalid.
        """
        if self.requested_timeframes is not None:
            valid_timeframes = ("1m", "5m", "15m", "1h", "4h")
            for tf in self.requested_timeframes:
                if tf not in valid_timeframes:
                    raise ValueError(f"requested_timeframes contains invalid timeframe: {tf}")
        if self.requested_horizons is not None:
            for h in self.requested_horizons:
                if h <= 0:
                    raise ValueError(f"requested_horizons must be positive, got: {h}")
        return self


class PredictionRecordContract(ContractModel):
    """Shared prediction record contract."""

    request_id: NonEmptyStr
    reference_ts: datetime
    horizon: PositiveInt
    event_probability: Probability
    confidence: Probability
    low_price: StrictFloat
    high_price: StrictFloat
    start_estimate: PositiveInt | None = None
    maturity_estimate: PositiveInt | None = None
    duration_estimate: PositiveInt | None = None
    low_confidence_advisory: StrictBool
    timeframe: Literal["1m", "5m", "15m", "1h", "4h"] | None = None

    @field_validator("reference_ts")
    @classmethod
    def validate_reference_ts(cls, value: datetime) -> datetime:
        """Require timezone-aware reference timestamps.

        Args:
            value: Candidate timestamp.

        Returns:
            The validated timestamp.

        Raises:
            ValueError: If the timestamp is naive.
        """
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reference_ts must be timezone-aware.")
        return value

    @field_validator("low_price", "high_price")
    @classmethod
    def validate_finite_boundary(cls, value: float) -> float:
        """Require finite boundary values.

        Args:
            value: Candidate boundary value.

        Returns:
            The validated boundary value.

        Raises:
            ValueError: If the value is NaN or infinite.
        """
        if not math.isfinite(value):
            raise ValueError("boundary values must be finite numbers.")
        return value


class AnalogRecordContract(ContractModel):
    """Shared analog explanation contract."""

    analog_id: NonEmptyStr
    reference_ts: datetime
    distance: PositiveFloat
    outcome_summary: NonEmptyStr


class PredictionResponseContract(ContractModel):
    """Shared prediction response contract."""

    prediction: PredictionRecordContract
    top_k_analogs: tuple[AnalogRecordContract, ...]
    summary_statistics: dict[str, StrictFloat]
    grounded_natural_language_explanation: NonEmptyStr

    @field_validator("summary_statistics")
    @classmethod
    def validate_summary_statistics(cls, value: dict[str, float]) -> dict[str, float]:
        """Require non-empty finite summary statistics.

        Args:
            value: Candidate summary statistics.

        Returns:
            The validated statistics mapping.

        Raises:
            ValueError: If the mapping is empty or contains non-finite values.
        """
        if not value:
            raise ValueError("summary_statistics must not be empty.")
        for key, metric in value.items():
            if not key.strip():
                raise ValueError("summary_statistics keys must not be empty.")
            if not math.isfinite(metric):
                raise ValueError("summary_statistics values must be finite numbers.")
        return value


class HealthResponseContract(ContractModel):
    """Shared health response contract."""

    status: Literal["ok", "degraded", "unavailable"]
    model_available: StrictBool


class StructuredErrorDetail(ContractModel):
    """Structured validation error detail."""

    location: tuple[NonEmptyStr, ...]
    message: NonEmptyStr
    input_value: Any | None = None

