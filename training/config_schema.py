"""Typed configuration schema for Sebuleni Pro Max AI.

This module defines the immutable runtime configuration contract required by U1.
All models are strict about unknown keys and are frozen after validation so the
same validated configuration can be shared across all runtime entry points.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

ACTIVE_TARGET_TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "1h", "4h", "1d")
TIME_PATTERN = re.compile(r"^\d{2}:\d{2}$")

NonEmptyStr = Annotated[StrictStr, Field(min_length=1)]
PositiveFloat = Annotated[StrictFloat, Field(gt=0.0)]
PortNumber = Annotated[StrictInt, Field(ge=1, le=65535)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]


class FrozenModel(BaseModel):
    """Base model with strict unknown-key rejection and immutability."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InstrumentConfig(FrozenModel):
    """Configuration for the active instrument."""

    instrument_id: NonEmptyStr


class DataSourceConfig(FrozenModel):
    """Configuration for source data locations."""

    ohlc_path: Path

    @field_validator("ohlc_path")
    @classmethod
    def validate_ohlc_path(cls, value: Path) -> Path:
        """Ensure the OHLC path is not empty.

        Args:
            value: Candidate file path.

        Returns:
            The validated path.

        Raises:
            ValueError: If the path is empty.
        """
        if not str(value).strip():
            raise ValueError("data_source.ohlc_path must not be empty.")
        return value


class DailyCloseConfig(FrozenModel):
    """Configuration for the daily close definition."""

    time: NonEmptyStr
    timezone: NonEmptyStr

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        """Validate a deterministic HH:MM time string.

        Args:
            value: Candidate time string.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value does not match HH:MM format.
        """
        if not TIME_PATTERN.fullmatch(value):
            raise ValueError("daily_close.time must use HH:MM 24-hour format.")

        hour, minute = value.split(":")
        if int(hour) > 23 or int(minute) > 59:
            raise ValueError("daily_close.time must be a valid 24-hour time.")
        return value


class TimeConfig(FrozenModel):
    """Timezone and daily close settings."""

    source_timezone: NonEmptyStr
    runtime_timezone: NonEmptyStr
    daily_close: DailyCloseConfig


class SessionDefinition(FrozenModel):
    """Definition of a single configured session window."""

    name: NonEmptyStr
    start: NonEmptyStr
    end: NonEmptyStr

    @field_validator("start", "end")
    @classmethod
    def validate_session_time(cls, value: str) -> str:
        """Validate session times.

        Args:
            value: Candidate session time string.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value does not match HH:MM format.
        """
        if not TIME_PATTERN.fullmatch(value):
            raise ValueError("session times must use HH:MM 24-hour format.")
        return value


class SessionsConfig(FrozenModel):
    """Session calendar and policy configuration."""

    calendar_name: NonEmptyStr
    weekend_policy: Literal["include", "exclude"]
    holiday_policy: Literal["include", "exclude"]
    definitions: tuple[SessionDefinition, ...]

    @field_validator("definitions")
    @classmethod
    def validate_definitions(cls, value: tuple[SessionDefinition, ...]) -> tuple[SessionDefinition, ...]:
        """Ensure at least one session definition exists.

        Args:
            value: Configured session definitions.

        Returns:
            The validated tuple.

        Raises:
            ValueError: If no session definitions are provided.
        """
        if not value:
            raise ValueError("sessions.definitions must contain at least one session definition.")
        return value


class ResamplingConfig(FrozenModel):
    """Resampling configuration for the active timeframe stack."""

    base_timeframe: Literal["1m"]
    target_timeframes: tuple[Literal["5m", "15m", "1h", "4h", "1d"], ...]

    @field_validator("target_timeframes")
    @classmethod
    def validate_target_timeframes(
        cls,
        value: tuple[Literal["5m", "15m", "1h", "4h", "1d"], ...],
    ) -> tuple[Literal["5m", "15m", "1h", "4h", "1d"], ...]:
        """Validate the approved derived timeframe set.

        Args:
            value: Configured target timeframes.

        Returns:
            The validated tuple.

        Raises:
            ValueError: If the configured timeframes do not match the approved
                active stack for derived frames.
        """
        if value != ACTIVE_TARGET_TIMEFRAMES:
            joined = ", ".join(ACTIVE_TARGET_TIMEFRAMES)
            raise ValueError(
                "resampling.target_timeframes must exactly match the approved "
                f"derived timeframe stack: {joined}."
            )
        return value


class FeaturesConfig(FrozenModel):
    """Feature toggle configuration."""

    enabled_features: tuple[NonEmptyStr, ...]
    deterministic_derived_features: tuple[NonEmptyStr, ...]

    @field_validator("enabled_features", "deterministic_derived_features")
    @classmethod
    def validate_feature_lists(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Ensure configured feature lists are non-empty and unique.

        Args:
            value: Candidate feature tuple.

        Returns:
            The validated tuple.

        Raises:
            ValueError: If the tuple is empty or contains duplicates.
        """
        if not value:
            raise ValueError("feature configuration lists must not be empty.")
        if len(set(value)) != len(value):
            raise ValueError("feature configuration lists must not contain duplicates.")
        return value


class LabelingConfig(FrozenModel):
    """Threshold and horizon configuration."""

    thresholds: tuple[PositiveFloat, ...]
    horizon_bars: tuple[PositiveInt, ...]

    @field_validator("thresholds", "horizon_bars")
    @classmethod
    def validate_non_empty_sequence(cls, value: tuple[object, ...]) -> tuple[object, ...]:
        """Ensure threshold and horizon lists are not empty.

        Args:
            value: Candidate sequence.

        Returns:
            The validated tuple.

        Raises:
            ValueError: If the sequence is empty.
        """
        if not value:
            raise ValueError("threshold and horizon sequences must not be empty.")
        return value


class WalkForwardConfig(FrozenModel):
    """Walk-forward split configuration."""

    train_bars: PositiveInt
    validation_bars: PositiveInt
    test_bars: PositiveInt
    step_bars: PositiveInt


class TrainingConfig(FrozenModel):
    """Training and runtime execution configuration."""

    random_seed: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    max_epochs: PositiveInt
    device_preference: Literal["cpu", "cuda"]
    allow_nondeterministic: StrictBool


class RetrievalConfig(FrozenModel):
    """Retrieval and explanation configuration."""

    top_k_analogs: PositiveInt


class ApiConfig(FrozenModel):
    """API configuration."""

    host: NonEmptyStr
    port: PortNumber


class ReportingConfig(FrozenModel):
    """Reporting configuration."""

    output_dir: Path

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value: Path) -> Path:
        """Ensure the output directory is not empty.

        Args:
            value: Candidate output directory.

        Returns:
            The validated directory path.

        Raises:
            ValueError: If the path is empty.
        """
        if not str(value).strip():
            raise ValueError("reporting.output_dir must not be empty.")
        return value


class SchedulesConfig(FrozenModel):
    """Schedule configuration."""

    reporting_cron: NonEmptyStr
    retraining_review_cron: NonEmptyStr


class RuntimeConfig(FrozenModel):
    """Top-level immutable runtime configuration."""

    instrument: InstrumentConfig
    data_source: DataSourceConfig
    time: TimeConfig
    sessions: SessionsConfig
    resampling: ResamplingConfig
    features: FeaturesConfig
    labeling: LabelingConfig
    walk_forward: WalkForwardConfig
    training: TrainingConfig
    retrieval: RetrievalConfig
    api: ApiConfig
    reporting: ReportingConfig
    schedules: SchedulesConfig

