"""U8 latent export for training-only retrieval memory."""

from __future__ import annotations

import logging
import math
from datetime import datetime

import numpy as np

from models.explanation import ExplanationError, RetrievalMemoryRecord

LOGGER = logging.getLogger(__name__)


class LatentExportError(ExplanationError):
    """Raised when retrieval-memory export fails."""


def export_training_latents(
    *,
    latent_matrix: np.ndarray,
    reference_ts: tuple[datetime, ...] | list[datetime],
    event_observed: tuple[float, ...] | list[float] | np.ndarray,
    future_low: tuple[float, ...] | list[float] | np.ndarray,
    future_high: tuple[float, ...] | list[float] | np.ndarray,
    event_start_offset: tuple[int | None, ...] | list[int | None],
    maturity_offset: tuple[int | None, ...] | list[int | None],
    source_fold_id: str,
    source_split: str,
) -> tuple[RetrievalMemoryRecord, ...]:
    """Export deterministic training-only retrieval memory rows.

    Args:
        latent_matrix: Latent matrix of shape ``(n_rows, latent_dim)``.
        reference_ts: Reference timestamps aligned with rows.
        event_observed: Event labels in ``[0, 1]``.
        future_low: Future low boundary values.
        future_high: Future high boundary values.
        event_start_offset: Event-start offsets.
        maturity_offset: Maturity offsets.
        source_fold_id: Fold identifier for audit purposes.
        source_split: Source split name. Must be ``"train"``.

    Returns:
        Immutable training-only retrieval memory rows.

    Raises:
        LatentExportError: If export inputs are invalid.
    """
    if source_split != "train":
        raise LatentExportError("retrieval memory export is allowed only for the train split.")
    if not source_fold_id.strip():
        raise LatentExportError("source_fold_id must not be empty.")

    matrix = np.asarray(latent_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise LatentExportError("latent_matrix must have shape (n_rows, latent_dim) with positive sizes.")
    if not np.isfinite(matrix).all():
        raise LatentExportError("latent_matrix must contain only finite values.")

    ts_values = tuple(reference_ts)
    event_values = _coerce_numeric_vector(event_observed, field_name="event_observed")
    low_values = _coerce_numeric_vector(future_low, field_name="future_low")
    high_values = _coerce_numeric_vector(future_high, field_name="future_high")
    start_values = tuple(event_start_offset)
    maturity_values = tuple(maturity_offset)

    row_count = matrix.shape[0]
    _validate_lengths(
        row_count=row_count,
        reference_ts=ts_values,
        event_observed=event_values,
        future_low=low_values,
        future_high=high_values,
        event_start_offset=start_values,
        maturity_offset=maturity_values,
    )

    rows: list[RetrievalMemoryRecord] = []
    for index in range(row_count):
        ts_value = ts_values[index]
        if ts_value.tzinfo is None or ts_value.utcoffset() is None:
            raise LatentExportError("reference_ts values must be timezone-aware.")

        event_value = float(event_values[index])
        if not 0.0 <= event_value <= 1.0:
            raise LatentExportError("event_observed values must lie in [0, 1].")

        low_value = float(low_values[index])
        high_value = float(high_values[index])
        if low_value > high_value:
            raise LatentExportError("future_low must not exceed future_high.")

        start_value = _normalize_optional_offset(start_values[index])
        maturity_value = _normalize_optional_offset(maturity_values[index])
        duration_value = _resolve_duration(start_value=start_value, maturity_value=maturity_value)

        outcome_summary = _build_outcome_summary(
            event_observed=event_value,
            future_low=low_value,
            future_high=high_value,
            duration_bars=duration_value,
        )
        rows.append(
            RetrievalMemoryRecord(
                analog_id=f"{source_fold_id}:{ts_value.isoformat()}",
                reference_ts=ts_value,
                latent_vector=tuple(float(value) for value in matrix[index]),
                outcome_summary=outcome_summary,
                event_observed=event_value,
                future_low=low_value,
                future_high=high_value,
                duration_bars=duration_value,
                source_split=source_split,
                source_fold_id=source_fold_id,
            )
        )

    LOGGER.info(
        "exported_training_latents",
        extra={
            "event": "exported_training_latents",
            "row_count": row_count,
            "latent_dim": int(matrix.shape[1]),
            "source_fold_id": source_fold_id,
            "source_split": source_split,
        },
    )
    return tuple(rows)


def _coerce_numeric_vector(values: tuple[float, ...] | list[float] | np.ndarray, *, field_name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise LatentExportError(f"{field_name} must be one-dimensional.")
    if not np.isfinite(array).all():
        raise LatentExportError(f"{field_name} must contain only finite values.")
    return tuple(float(item) for item in array)


def _validate_lengths(
    *,
    row_count: int,
    reference_ts: tuple[datetime, ...],
    event_observed: tuple[float, ...],
    future_low: tuple[float, ...],
    future_high: tuple[float, ...],
    event_start_offset: tuple[int | None, ...],
    maturity_offset: tuple[int | None, ...],
) -> None:
    if len(reference_ts) != row_count:
        raise LatentExportError("reference_ts length must match latent_matrix row count.")
    if len(event_observed) != row_count:
        raise LatentExportError("event_observed length must match latent_matrix row count.")
    if len(future_low) != row_count:
        raise LatentExportError("future_low length must match latent_matrix row count.")
    if len(future_high) != row_count:
        raise LatentExportError("future_high length must match latent_matrix row count.")
    if len(event_start_offset) != row_count:
        raise LatentExportError("event_start_offset length must match latent_matrix row count.")
    if len(maturity_offset) != row_count:
        raise LatentExportError("maturity_offset length must match latent_matrix row count.")


def _normalize_optional_offset(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 0:
        return None
    return int(value)


def _resolve_duration(*, start_value: int | None, maturity_value: int | None) -> float | None:
    if start_value is None or maturity_value is None:
        return None
    duration = (maturity_value - start_value) + 1
    if duration <= 0:
        raise LatentExportError("duration must be positive when both offsets are provided.")
    return float(duration)


def _build_outcome_summary(
    *,
    event_observed: float,
    future_low: float,
    future_high: float,
    duration_bars: float | None,
) -> str:
    if not math.isfinite(event_observed) or not math.isfinite(future_low) or not math.isfinite(future_high):
        raise LatentExportError("outcome summary inputs must be finite.")
    duration_text = "na" if duration_bars is None else f"{duration_bars:.1f}"
    return (
        f"event_observed={event_observed:.0f}; "
        f"future_low={future_low:.4f}; "
        f"future_high={future_high:.4f}; "
        f"duration_bars={duration_text}"
    )
