"""U12 operational monitoring and alert evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class MonitoringError(ValueError):
    """Raised when monitoring inputs are invalid."""


@dataclass(frozen=True)
class MonitoringThresholds:
    """Threshold configuration for live-health evaluation."""

    max_latency_ms: float
    max_low_confidence_rate: float
    max_report_age_minutes: float


@dataclass(frozen=True)
class MonitoringAlert:
    """Structured operational alert."""

    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class LiveHealthEvaluation:
    """Typed live-health result."""

    status: str
    metrics: dict[str, float]
    alerts: tuple[MonitoringAlert, ...]


def evaluate_live_health(
    *,
    latency_ms: tuple[float, ...] | list[float],
    low_confidence_flags: tuple[bool, ...] | list[bool],
    active_model_manifest_path: str | Path,
    latest_report_path: str | Path,
    thresholds: MonitoringThresholds,
    current_time: datetime | None = None,
) -> LiveHealthEvaluation:
    """Evaluate operational health from latency, advisories, and artifact freshness."""
    _validate_thresholds(thresholds)
    latency_values = tuple(float(item) for item in latency_ms)
    if not latency_values:
        raise MonitoringError("latency_ms must not be empty.")
    if any(item < 0.0 for item in latency_values):
        raise MonitoringError("latency_ms values must be non-negative.")

    advisory_flags = tuple(bool(item) for item in low_confidence_flags)
    if len(advisory_flags) != len(latency_values):
        raise MonitoringError("low_confidence_flags must align with latency_ms.")

    manifest_path = Path(active_model_manifest_path)
    report_path = Path(latest_report_path)
    effective_now = current_time or datetime.now(timezone.utc)
    if effective_now.tzinfo is None or effective_now.utcoffset() is None:
        raise MonitoringError("current_time must be timezone-aware when provided.")

    alerts: list[MonitoringAlert] = []
    if not manifest_path.exists():
        alerts.append(
            MonitoringAlert(
                code="missing_active_model_manifest",
                severity="critical",
                message=f"Active model manifest is missing: {manifest_path}",
            )
        )
    if not report_path.exists():
        alerts.append(
            MonitoringAlert(
                code="missing_report_artifact",
                severity="warning",
                message=f"Latest scheduled report is missing: {report_path}",
            )
        )

    mean_latency = sum(latency_values) / len(latency_values)
    low_confidence_rate = sum(1 for item in advisory_flags if item) / len(advisory_flags)
    report_age_minutes = _resolve_report_age_minutes(report_path=report_path, current_time=effective_now)

    if mean_latency > thresholds.max_latency_ms:
        alerts.append(
            MonitoringAlert(
                code="latency_threshold_breach",
                severity="warning",
                message=(
                    f"Mean latency {mean_latency:.4f}ms exceeds threshold "
                    f"{thresholds.max_latency_ms:.4f}ms."
                ),
            )
        )
    if low_confidence_rate > thresholds.max_low_confidence_rate:
        alerts.append(
            MonitoringAlert(
                code="advisory_rate_threshold_breach",
                severity="warning",
                message=(
                    f"Low-confidence advisory rate {low_confidence_rate:.4f} exceeds threshold "
                    f"{thresholds.max_low_confidence_rate:.4f}."
                ),
            )
        )
    if report_age_minutes > thresholds.max_report_age_minutes:
        alerts.append(
            MonitoringAlert(
                code="stale_report_artifact",
                severity="warning",
                message=(
                    f"Latest report age {report_age_minutes:.4f} minutes exceeds threshold "
                    f"{thresholds.max_report_age_minutes:.4f}."
                ),
            )
        )

    status = "ok" if not alerts else ("degraded" if all(alert.severity != "critical" for alert in alerts) else "unavailable")
    metrics = {
        "mean_latency_ms": round(mean_latency, 8),
        "low_confidence_rate": round(low_confidence_rate, 8),
        "report_age_minutes": round(report_age_minutes, 8),
        "alert_count": float(len(alerts)),
    }

    LOGGER.info(
        "evaluated_live_health",
        extra={
            "event": "evaluated_live_health",
            "status": status,
            "mean_latency_ms": metrics["mean_latency_ms"],
            "low_confidence_rate": metrics["low_confidence_rate"],
            "report_age_minutes": metrics["report_age_minutes"],
            "alert_codes": [alert.code for alert in alerts],
        },
    )
    return LiveHealthEvaluation(status=status, metrics=metrics, alerts=tuple(alerts))


def _resolve_report_age_minutes(*, report_path: Path, current_time: datetime) -> float:
    if not report_path.exists():
        return float("inf")
    modified_at = datetime.fromtimestamp(report_path.stat().st_mtime, tz=timezone.utc)
    return max(0.0, (current_time - modified_at).total_seconds() / 60.0)


def _validate_thresholds(thresholds: MonitoringThresholds) -> None:
    if thresholds.max_latency_ms <= 0.0:
        raise MonitoringError("max_latency_ms must be positive.")
    if not 0.0 <= thresholds.max_low_confidence_rate <= 1.0:
        raise MonitoringError("max_low_confidence_rate must lie in [0, 1].")
    if thresholds.max_report_age_minutes <= 0.0:
        raise MonitoringError("max_report_age_minutes must be positive.")
