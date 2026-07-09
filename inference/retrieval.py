"""Training-only deterministic analog retrieval for U8."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from models.explanation import (
    ExplanationError,
    RetrievalAudit,
    RetrievalEvidence,
    RetrievalIndex,
    RetrievalMemoryRecord,
    validate_memory_record,
)
from training.contracts import AnalogRecordContract

LOGGER = logging.getLogger(__name__)


def build_retrieval_index(memory_rows: tuple[RetrievalMemoryRecord, ...] | list[RetrievalMemoryRecord]) -> RetrievalIndex:
    """Build a deterministic training-only retrieval index.

    Args:
        memory_rows: Training-only retrieval memory rows.

    Returns:
        Deterministic in-memory retrieval index.

    Raises:
        ExplanationError: If the memory rows are invalid.
    """
    rows = tuple(memory_rows)
    if not rows:
        raise ExplanationError("retrieval memory must not be empty.")

    for row in rows:
        validate_memory_record(row)

    dimension = len(rows[0].latent_vector)
    for row in rows:
        if len(row.latent_vector) != dimension:
            raise ExplanationError("all latent vectors in retrieval memory must have the same dimension.")

    matrix = np.asarray([row.latent_vector for row in rows], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ExplanationError("retrieval memory matrix must have shape (n, latent_dim).")
    if not np.isfinite(matrix).all():
        raise ExplanationError("retrieval memory matrix must contain only finite values.")

    LOGGER.info(
        "built_retrieval_index",
        extra={
            "event": "built_retrieval_index",
            "row_count": int(matrix.shape[0]),
            "dimension": int(matrix.shape[1]),
            "index_scope": "train_only",
        },
    )
    return RetrievalIndex(records=rows, vectors=matrix, dimension=dimension)


def retrieve_analogs(
    index: RetrievalIndex,
    query_latent: np.ndarray | tuple[float, ...] | list[float],
    *,
    query_reference_ts: datetime,
    top_k: int,
) -> RetrievalEvidence:
    """Retrieve deterministic top-K analogs from a training-only index.

    Args:
        index: Deterministic retrieval index.
        query_latent: Query latent vector.
        query_reference_ts: Timestamp being explained.
        top_k: Number of analogs to retrieve.

    Returns:
        Retrieved evidence with analogs, summary statistics, and audit metadata.

    Raises:
        ExplanationError: If retrieval cannot be performed safely.
    """
    if top_k <= 0:
        raise ExplanationError("top_k must be positive.")
    if query_reference_ts.tzinfo is None or query_reference_ts.utcoffset() is None:
        raise ExplanationError("query_reference_ts must be timezone-aware.")

    latent = np.asarray(query_latent, dtype=np.float64)
    if latent.ndim != 1:
        raise ExplanationError("query_latent must have shape (latent_dim,).")
    if latent.shape[0] != index.dimension:
        raise ExplanationError(
            f"query_latent dimension mismatch: expected {index.dimension}, got {latent.shape[0]}."
        )
    if not np.isfinite(latent).all():
        raise ExplanationError("query_latent must contain only finite values.")

    eligible: list[tuple[int, RetrievalMemoryRecord]] = []
    filtered_future_count = 0
    for idx, row in enumerate(index.records):
        if row.reference_ts >= query_reference_ts:
            filtered_future_count += 1
            continue
        eligible.append((idx, row))

    if not eligible:
        raise ExplanationError("no training analogs remain after applying chronology filtering.")

    ranked = []
    for idx, row in eligible:
        distance = float(np.linalg.norm(index.vectors[idx] - latent))
        if distance <= 0.0:
            distance = float(np.finfo(np.float64).eps)
        ranked.append((distance, row.reference_ts.isoformat(), row.analog_id, row))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = ranked[:top_k]

    analogs = tuple(
        AnalogRecordContract(
            analog_id=row.analog_id,
            reference_ts=row.reference_ts,
            distance=float(distance),
            outcome_summary=row.outcome_summary,
        )
        for distance, _, _, row in selected
    )
    statistics = _build_summary_statistics(selected)
    audit = RetrievalAudit(
        index_scope="train_only",
        query_reference_ts=query_reference_ts,
        candidate_count=len(index.records),
        filtered_future_count=filtered_future_count,
        returned_count=len(analogs),
    )

    LOGGER.info(
        "retrieved_analogs",
        extra={
            "event": "retrieved_analogs",
            "candidate_count": len(index.records),
            "filtered_future_count": filtered_future_count,
            "returned_count": len(analogs),
            "top_k": int(top_k),
            "query_reference_ts": query_reference_ts.isoformat(),
        },
    )
    return RetrievalEvidence(
        analogs=analogs,
        summary_statistics=statistics,
        audit=audit,
    )


def _build_summary_statistics(
    ranked_rows: list[tuple[float, str, str, RetrievalMemoryRecord]],
) -> dict[str, float]:
    if not ranked_rows:
        raise ExplanationError("ranked_rows must not be empty when computing summary statistics.")

    distances = np.asarray([item[0] for item in ranked_rows], dtype=np.float64)
    event_observed = np.asarray([item[3].event_observed for item in ranked_rows], dtype=np.float64)
    future_lows = np.asarray([item[3].future_low for item in ranked_rows], dtype=np.float64)
    future_highs = np.asarray([item[3].future_high for item in ranked_rows], dtype=np.float64)
    durations = np.asarray(
        [item[3].duration_bars for item in ranked_rows if item[3].duration_bars is not None],
        dtype=np.float64,
    )
    if durations.size == 0:
        durations = np.asarray([0.0], dtype=np.float64)

    return {
        "analog_count": float(len(ranked_rows)),
        "mean_distance": float(np.mean(distances)),
        "observed_event_rate": float(np.mean(event_observed)),
        "mean_future_low": float(np.mean(future_lows)),
        "mean_future_high": float(np.mean(future_highs)),
        "mean_boundary_span": float(np.mean(future_highs - future_lows)),
        "mean_duration_bars": float(np.mean(durations)),
    }
