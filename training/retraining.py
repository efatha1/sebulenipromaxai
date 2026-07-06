"""U12 retraining request queue management."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class RetrainingError(ValueError):
    """Raised when retraining queue operations fail."""


@dataclass(frozen=True)
class RetrainingRequestRecord:
    """Typed retraining request record."""

    request_id: str
    instrument_id: str
    candidate_model_id: str | None
    reason: str
    requested_at: datetime
    status: str
    source: str
    output_path: Path


def request_retraining(
    *,
    output_dir: str | Path,
    instrument_id: str,
    reason: str,
    candidate_model_id: str | None = None,
    source: str = "manual",
    requested_at: datetime | None = None,
) -> RetrainingRequestRecord:
    """Create a retraining request artifact for manual or scheduled review."""
    if not instrument_id.strip():
        raise RetrainingError("instrument_id must not be empty.")
    if not reason.strip():
        raise RetrainingError("reason must not be empty.")
    if source not in {"manual", "scheduled"}:
        raise RetrainingError("source must be either 'manual' or 'scheduled'.")

    ts = requested_at or datetime.now(timezone.utc)
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise RetrainingError("requested_at must be timezone-aware when provided.")

    request_id = f"rr-{ts.strftime('%Y%m%d%H%M%S')}-{source}"
    path = Path(output_dir) / "retraining_requests" / f"{request_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "request_id": request_id,
        "instrument_id": instrument_id,
        "candidate_model_id": candidate_model_id,
        "reason": reason,
        "requested_at": ts.isoformat(),
        "status": "pending_review",
        "source": source,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    LOGGER.info(
        "created_retraining_request",
        extra={
            "event": "created_retraining_request",
            "request_id": request_id,
            "instrument_id": instrument_id,
            "source": source,
            "output_path": str(path),
        },
    )
    return RetrainingRequestRecord(
        request_id=request_id,
        instrument_id=instrument_id,
        candidate_model_id=candidate_model_id,
        reason=reason,
        requested_at=ts,
        status="pending_review",
        source=source,
        output_path=path,
    )


def load_retraining_requests(
    output_dir: str | Path,
    *,
    statuses: tuple[str, ...] | list[str] | None = None,
) -> tuple[RetrainingRequestRecord, ...]:
    """Load retraining requests from the configured output directory."""
    root = Path(output_dir) / "retraining_requests"
    if not root.exists():
        return tuple()
    allowed_statuses = set(statuses) if statuses is not None else None
    records: list[RetrainingRequestRecord] = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = _deserialize_request(payload=payload, path=path)
        if allowed_statuses is not None and record.status not in allowed_statuses:
            continue
        records.append(record)
    return tuple(records)


def update_retraining_request_status(
    request_path: str | Path,
    *,
    status: str,
    reviewed_at: datetime | None = None,
    review_note: str | None = None,
) -> RetrainingRequestRecord:
    """Update the status of a retraining request in place."""
    path = Path(request_path)
    if not path.exists():
        raise RetrainingError(f"retraining request does not exist: {path}")
    if status not in {"pending_review", "reviewed", "approved", "rejected"}:
        raise RetrainingError("unsupported retraining request status.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = status
    if reviewed_at is not None:
        if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
            raise RetrainingError("reviewed_at must be timezone-aware when provided.")
        payload["reviewed_at"] = reviewed_at.isoformat()
    if review_note is not None:
        payload["review_note"] = review_note
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    LOGGER.info(
        "updated_retraining_request_status",
        extra={
            "event": "updated_retraining_request_status",
            "request_path": str(path),
            "status": status,
        },
    )
    return _deserialize_request(payload=json.loads(path.read_text(encoding="utf-8")), path=path)


def _deserialize_request(*, payload: dict[str, Any], path: Path) -> RetrainingRequestRecord:
    required = {"request_id", "instrument_id", "reason", "requested_at", "status"}
    missing = sorted(required.difference(payload))
    if missing:
        raise RetrainingError(f"retraining request missing required fields: {missing}")
    source = str(payload.get("source", "manual"))
    return RetrainingRequestRecord(
        request_id=str(payload["request_id"]),
        instrument_id=str(payload["instrument_id"]),
        candidate_model_id=(str(payload["candidate_model_id"]) if payload.get("candidate_model_id") else None),
        reason=str(payload["reason"]),
        requested_at=datetime.fromisoformat(str(payload["requested_at"])),
        status=str(payload["status"]),
        source=source,
        output_path=path,
    )
