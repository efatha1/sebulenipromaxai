"""Unit tests for U1 API schemas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from api.schemas import PredictRequest, PredictResponse


def build_bar(timestamp: datetime) -> dict[str, object]:
    """Build a valid OHLC bar payload."""
    return {
        "timestamp": timestamp.isoformat(),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
    }


def test_predict_request_accepts_valid_payload() -> None:
    """A valid predict request should validate successfully."""
    payload = {
        "instrument_id": "TEST_INSTRUMENT",
        "bars_1m": [
            build_bar(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)),
            build_bar(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)),
        ],
        "horizon_mode": "single",
        "horizon_bars": 15,
        "threshold": 10.0,
        "top_k_analogs": 5,
    }

    request = PredictRequest.model_validate(payload)

    assert request.horizon_mode == "single"
    assert len(request.bars_1m) == 2


def test_predict_request_rejects_duplicate_timestamps() -> None:
    """Public bars should reject duplicate timestamps."""
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    payload = {
        "instrument_id": "TEST_INSTRUMENT",
        "bars_1m": [build_bar(ts), build_bar(ts)],
        "horizon_mode": "single",
        "horizon_bars": 15,
        "threshold": 10.0,
        "top_k_analogs": 5,
    }

    with pytest.raises(ValidationError, match="duplicate"):
        PredictRequest.model_validate(payload)


def test_predict_request_requires_horizon_bars_in_single_mode() -> None:
    """Single horizon mode requires horizon_bars."""
    payload = {
        "instrument_id": "TEST_INSTRUMENT",
        "bars_1m": [build_bar(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))],
        "horizon_mode": "single",
        "threshold": 10.0,
        "top_k_analogs": 5,
    }

    with pytest.raises(ValidationError, match="horizon_bars"):
        PredictRequest.model_validate(payload)


def test_predict_response_requires_analogs() -> None:
    """Prediction responses should require analog evidence records."""
    payload = {
        "prediction": {
            "request_id": "req-1",
            "reference_ts": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat(),
            "horizon": 15,
            "event_probability": 0.9,
            "confidence": 0.8,
            "low_price": 95.0,
            "high_price": 105.0,
            "start_estimate": 3,
            "maturity_estimate": 9,
            "duration_estimate": 6,
            "low_confidence_advisory": False,
        },
        "top_k_analogs": [],
        "summary_statistics": {"mean_distance": 0.12},
        "grounded_natural_language_explanation": "Grounded in similar historical states.",
    }

    with pytest.raises(ValidationError, match="analog"):
        PredictResponse.model_validate(payload)

