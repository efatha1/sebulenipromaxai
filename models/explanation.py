"""U8 explanation retrieval contracts and grounded rendering."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from inference.explanation_templates import build_grounded_explanation_text
from training.contracts import AnalogRecordContract, PredictionRecordContract

LOGGER = logging.getLogger(__name__)


class ExplanationError(ValueError):
    """Raised when explanation retrieval or rendering fails."""


@dataclass(frozen=True)
class RetrievalMemoryRecord:
    """Immutable training-only retrieval memory row."""

    analog_id: str
    reference_ts: datetime
    latent_vector: tuple[float, ...]
    outcome_summary: str
    event_observed: float
    future_low: float
    future_high: float
    duration_bars: float | None
    source_split: str
    source_fold_id: str


@dataclass(frozen=True)
class RetrievalIndex:
    """Deterministic in-memory retrieval index."""

    records: tuple[RetrievalMemoryRecord, ...]
    vectors: np.ndarray
    dimension: int


@dataclass(frozen=True)
class RetrievalAudit:
    """Audit metadata for explanation provenance."""

    index_scope: str
    query_reference_ts: datetime
    candidate_count: int
    filtered_future_count: int
    returned_count: int


@dataclass(frozen=True)
class RetrievalEvidence:
    """Retrieved analog evidence and derived summary statistics."""

    analogs: tuple[AnalogRecordContract, ...]
    summary_statistics: dict[str, float]
    audit: RetrievalAudit


@dataclass(frozen=True)
class GroundedExplanation:
    """Typed grounded explanation output."""

    top_k_analogs: tuple[AnalogRecordContract, ...]
    summary_statistics: dict[str, float]
    grounded_natural_language_explanation: str
    audit: RetrievalAudit


def render_explanation(
    prediction: PredictionRecordContract,
    evidence: RetrievalEvidence,
    *,
    requested_top_k: int,
) -> GroundedExplanation:
    """Render a deterministic grounded explanation.

    Args:
        prediction: Typed prediction payload from `U7` or later inference assembly.
        evidence: Retrieved analog evidence.
        requested_top_k: Requested analog count from configuration or request.

    Returns:
        Typed grounded explanation output.

    Raises:
        ExplanationError: If the explanation cannot be grounded safely.
    """
    if requested_top_k <= 0:
        raise ExplanationError("requested_top_k must be positive.")
    if prediction.reference_ts.tzinfo is None or prediction.reference_ts.utcoffset() is None:
        raise ExplanationError("prediction.reference_ts must be timezone-aware.")
    if evidence.audit.index_scope != "train_only":
        raise ExplanationError("explanations must be grounded on a train_only retrieval index.")
    if not evidence.analogs:
        raise ExplanationError("at least one analog is required to render an explanation.")
    if not evidence.summary_statistics:
        raise ExplanationError("summary_statistics must not be empty.")

    analog_count = int(evidence.summary_statistics.get("analog_count", 0.0))
    if analog_count != len(evidence.analogs):
        raise ExplanationError("analog_count statistic must match the number of retrieved analogs.")
    if prediction.confidence > 0.75 and analog_count == 0:
        raise ExplanationError("high-confidence predictions must not be unexplained.")

    explanation = build_grounded_explanation_text(
        prediction=prediction,
        analogs=evidence.analogs,
        summary_statistics=evidence.summary_statistics,
        requested_top_k=requested_top_k,
    )
    if not explanation.strip():
        raise ExplanationError("grounded explanation text must not be empty.")
    if prediction.confidence > 0.75 and "analog" not in explanation.lower():
        raise ExplanationError("high-confidence explanations must explicitly reference analog evidence.")

    LOGGER.info(
        "rendered_grounded_explanation",
        extra={
            "event": "rendered_grounded_explanation",
            "reference_ts": prediction.reference_ts.isoformat(),
            "analog_count": analog_count,
            "requested_top_k": int(requested_top_k),
            "low_confidence_advisory": bool(prediction.low_confidence_advisory),
        },
    )
    return GroundedExplanation(
        top_k_analogs=evidence.analogs,
        summary_statistics=dict(evidence.summary_statistics),
        grounded_natural_language_explanation=explanation,
        audit=evidence.audit,
    )


def validate_memory_record(record: RetrievalMemoryRecord) -> None:
    """Validate a retrieval memory record.

    Args:
        record: Candidate retrieval memory record.

    Raises:
        ExplanationError: If the record is malformed.
    """
    if not record.analog_id.strip():
        raise ExplanationError("analog_id must not be empty.")
    if record.reference_ts.tzinfo is None or record.reference_ts.utcoffset() is None:
        raise ExplanationError("reference_ts must be timezone-aware.")
    if not record.outcome_summary.strip():
        raise ExplanationError("outcome_summary must not be empty.")
    if record.source_split != "train":
        raise ExplanationError("retrieval memory records must come from the train split only.")
    if not record.source_fold_id.strip():
        raise ExplanationError("source_fold_id must not be empty.")
    if not record.latent_vector:
        raise ExplanationError("latent_vector must not be empty.")
    for value in record.latent_vector:
        if not math.isfinite(value):
            raise ExplanationError("latent_vector must contain only finite values.")
    for value_name in ("event_observed", "future_low", "future_high"):
        value = getattr(record, value_name)
        if not math.isfinite(value):
            raise ExplanationError(f"{value_name} must be finite.")
    if not 0.0 <= record.event_observed <= 1.0:
        raise ExplanationError("event_observed must lie in [0, 1].")
    if record.future_low > record.future_high:
        raise ExplanationError("future_low must not exceed future_high.")
    if record.duration_bars is not None and not math.isfinite(record.duration_bars):
        raise ExplanationError("duration_bars must be finite when provided.")
