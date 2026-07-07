"""API request and response schemas for Sebuleni Pro Max AI."""

from __future__ import annotations

from pydantic import field_validator, model_validator

from training.contracts import (
    AnalogRecordContract,
    HealthResponseContract,
    OhlcBarContract,
    PredictionRecordContract,
    PredictionRequestContract,
    PredictionResponseContract,
)


class PredictRequest(PredictionRequestContract):
    """REST request schema for `/predict`."""

    @field_validator("bars_1m")
    @classmethod
    def validate_chronological_bars(
        cls,
        value: tuple[OhlcBarContract, ...],
    ) -> tuple[OhlcBarContract, ...]:
        """Require strictly increasing unique timestamps.

        Args:
            value: Validated OHLC bars.

        Returns:
            The validated bar tuple.

        Raises:
            ValueError: If timestamps are duplicated or out of order.
        """
        timestamps = [bar.timestamp for bar in value]
        if timestamps != sorted(timestamps):
            raise ValueError("bars_1m must be strictly chronological.")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("bars_1m must not contain duplicate timestamps.")
        return value


class PredictResponse(PredictionResponseContract):
    """REST response schema for `/predict`."""

    @model_validator(mode="after")
    def validate_analog_count(self) -> "PredictResponse":
        """Require at least one analog when a response is emitted.

        Returns:
            The validated response.

        Raises:
            ValueError: If no analogs are present.
        """
        if not self.top_k_analogs:
            raise ValueError("top_k_analogs must contain at least one analog record.")
        return self


class HealthResponse(HealthResponseContract):
    """REST response schema for `/health`."""


__all__ = [
    "AnalogRecordContract",
    "HealthResponse",
    "OhlcBarContract",
    "PredictRequest",
    "PredictResponse",
    "PredictionRecordContract",
]

