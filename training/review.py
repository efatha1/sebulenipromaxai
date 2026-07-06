"""U12 review-gated approval and rollback metadata."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.dependencies import ActiveModelManifest, load_active_model_manifest, save_active_model_manifest
from models.explanation import RetrievalMemoryRecord
from training.evaluate import EvaluationSummary

LOGGER = logging.getLogger(__name__)


class ReviewError(ValueError):
    """Raised when controlled review operations fail."""


@dataclass(frozen=True)
class ReviewDecision:
    """Typed review decision and artifact metadata."""

    status: str
    review_artifact_path: Path
    active_model_manifest_path: Path
    rollback_manifest_path: Path | None
    candidate_model_id: str


def approve_candidate(
    *,
    candidate_summary: EvaluationSummary,
    active_model_manifest_path: str | Path,
    lookbacks_by_timeframe: dict[str, int],
    retrieval_memory: tuple[RetrievalMemoryRecord, ...] | list[RetrievalMemoryRecord],
    output_dir: str | Path,
    approved: bool,
    review_reason: str,
    reviewer_id: str,
    reviewed_at: datetime | None = None,
) -> ReviewDecision:
    """Approve or reject a candidate model with explicit review gating."""
    if not reviewer_id.strip():
        raise ReviewError("reviewer_id must not be empty.")
    if not review_reason.strip():
        raise ReviewError("review_reason must not be empty.")

    manifest_path = Path(active_model_manifest_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    decision_time = reviewed_at or datetime.now(timezone.utc)
    if decision_time.tzinfo is None or decision_time.utcoffset() is None:
        raise ReviewError("reviewed_at must be timezone-aware when provided.")

    candidate = candidate_summary.candidate_model
    if not candidate.artifact_path.exists():
        raise ReviewError(f"candidate artifact does not exist: {candidate.artifact_path}")

    current_manifest = load_active_model_manifest(manifest_path) if manifest_path.exists() else None
    review_dir = output_root / "review_decisions"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_artifact_path = review_dir / f"{candidate.model_id}-{decision_time.strftime('%Y%m%d%H%M%S')}.json"

    rollback_path: Path | None = None
    status = "approved" if approved else "rejected"
    if approved:
        rollback_path = _write_rollback_snapshot(
            output_root=output_root,
            current_manifest=current_manifest,
            candidate_model_id=candidate.model_id,
            decision_time=decision_time,
        )
        save_active_model_manifest(
            manifest_path,
            ActiveModelManifest(
                checkpoint_path=str(candidate.artifact_path),
                lookbacks_by_timeframe=dict(lookbacks_by_timeframe),
                retrieval_memory=tuple(retrieval_memory),
            ),
        )

    payload = {
        "status": status,
        "candidate_model_id": candidate.model_id,
        "candidate_artifact_path": str(candidate.artifact_path),
        "review_reason": review_reason,
        "reviewer_id": reviewer_id,
        "reviewed_at": decision_time.isoformat(),
        "rollback_manifest_path": str(rollback_path) if rollback_path is not None else None,
        "current_active_manifest_path": str(manifest_path),
        "approved": bool(approved),
    }
    review_artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    LOGGER.info(
        "review_decision_recorded",
        extra={
            "event": "review_decision_recorded",
            "status": status,
            "candidate_model_id": candidate.model_id,
            "review_artifact_path": str(review_artifact_path),
            "rollback_manifest_path": str(rollback_path) if rollback_path is not None else None,
        },
    )
    return ReviewDecision(
        status=status,
        review_artifact_path=review_artifact_path,
        active_model_manifest_path=manifest_path,
        rollback_manifest_path=rollback_path,
        candidate_model_id=candidate.model_id,
    )


def write_review_recommendation(
    *,
    output_dir: str | Path,
    candidate_summary: EvaluationSummary | None,
    request_ids: tuple[str, ...] | list[str],
    recommended_action: str,
    current_time: datetime | None = None,
    note: str,
) -> Path:
    """Write a review artifact without changing the active model."""
    if not recommended_action.strip():
        raise ReviewError("recommended_action must not be empty.")
    if not note.strip():
        raise ReviewError("note must not be empty.")
    ts = current_time or datetime.now(timezone.utc)
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ReviewError("current_time must be timezone-aware when provided.")

    review_dir = Path(output_dir) / "review_runs"
    review_dir.mkdir(parents=True, exist_ok=True)
    candidate_model_id = candidate_summary.candidate_model.model_id if candidate_summary is not None else None
    path = review_dir / f"scheduled-review-{ts.strftime('%Y%m%d%H%M%S')}.json"
    payload: dict[str, Any] = {
        "reviewed_at": ts.isoformat(),
        "request_ids": list(request_ids),
        "candidate_model_id": candidate_model_id,
        "recommended_action": recommended_action,
        "note": note,
        "auto_promotion_performed": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info(
        "scheduled_review_artifact_written",
        extra={
            "event": "scheduled_review_artifact_written",
            "review_artifact_path": str(path),
            "recommended_action": recommended_action,
            "request_count": len(tuple(request_ids)),
            "candidate_model_id": candidate_model_id,
        },
    )
    return path


def _write_rollback_snapshot(
    *,
    output_root: Path,
    current_manifest: ActiveModelManifest | None,
    candidate_model_id: str,
    decision_time: datetime,
) -> Path | None:
    if current_manifest is None:
        return None
    rollback_dir = output_root / "rollback_manifests"
    rollback_dir.mkdir(parents=True, exist_ok=True)
    path = rollback_dir / f"{candidate_model_id}-{decision_time.strftime('%Y%m%d%H%M%S')}.json"
    save_active_model_manifest(path, current_manifest)
    return path
