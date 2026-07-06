"""U9 training metrics helpers and structured metric logging."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.losses import AcceptanceMetrics, HeadLosses

LOGGER = logging.getLogger(__name__)


class MetricsError(ValueError):
    """Raised when U9 metrics handling fails."""


@dataclass(frozen=True)
class StructuredMetricRecord:
    """Structured metric log record."""

    run_id: str
    fold_id: int
    split_name: str
    sample_count: int
    metrics: dict[str, float]


def summarize_split_metrics(
    *,
    sample_count: int,
    losses: HeadLosses,
    acceptance_metrics: AcceptanceMetrics,
    explanation_coverage_rate: float | None = None,
) -> dict[str, float]:
    """Build a stable metrics dictionary for one split.

    Args:
        sample_count: Number of rows in the split.
        losses: Per-task and aggregate losses.
        acceptance_metrics: Offline acceptance-aligned metrics.
        explanation_coverage_rate: Optional explanation coverage metric.

    Returns:
        Stable dictionary of finite metrics.
    """
    if sample_count <= 0:
        raise MetricsError("sample_count must be positive.")

    metrics = {
        "sample_count": _stable_metric(sample_count),
        "total_loss": _stable_metric(losses.total_loss.detach().cpu().item()),
        "event_loss": _stable_metric(losses.event_loss.detach().cpu().item()),
        "boundary_loss": _stable_metric(losses.boundary_loss.detach().cpu().item()),
        "timing_loss": _stable_metric(losses.timing_loss.detach().cpu().item()),
        "confidence_loss": _stable_metric(losses.confidence_loss.detach().cpu().item()),
        "regime_loss": _stable_metric(losses.regime_loss.detach().cpu().item()),
        "event_brier": _stable_metric(acceptance_metrics.event_brier),
        "boundary_mae": _stable_metric(acceptance_metrics.boundary_mae),
    }
    metrics["timing_mae"] = (
        _stable_metric(acceptance_metrics.timing_mae) if acceptance_metrics.timing_mae is not None else -1.0
    )
    metrics["confidence_brier"] = (
        _stable_metric(acceptance_metrics.confidence_brier) if acceptance_metrics.confidence_brier is not None else -1.0
    )
    metrics["regime_cross_entropy"] = (
        _stable_metric(acceptance_metrics.regime_cross_entropy) if acceptance_metrics.regime_cross_entropy is not None else -1.0
    )
    if explanation_coverage_rate is not None:
        if not 0.0 <= explanation_coverage_rate <= 1.0:
            raise MetricsError("explanation_coverage_rate must lie in [0, 1].")
        metrics["explanation_coverage_rate"] = _stable_metric(explanation_coverage_rate)

    _validate_metric_mapping(metrics)
    return metrics


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]] | tuple[dict[str, float], ...]) -> dict[str, float]:
    """Average metrics across folds.

    Args:
        metric_dicts: Per-fold metric mappings with identical keys.

    Returns:
        Mean metric mapping.
    """
    metrics = list(metric_dicts)
    if not metrics:
        raise MetricsError("metric_dicts must not be empty.")

    keys = tuple(sorted(metrics[0].keys()))
    for mapping in metrics:
        if tuple(sorted(mapping.keys())) != keys:
            raise MetricsError("all metric mappings must have identical keys for aggregation.")
        _validate_metric_mapping(mapping)

    out: dict[str, float] = {}
    for key in keys:
        out[key] = float(sum(mapping[key] for mapping in metrics) / len(metrics))
    return out


def append_metric_record(log_path: str | Path, record: StructuredMetricRecord) -> None:
    """Append a structured metrics record to a JSONL log.

    Args:
        log_path: Destination JSONL file.
        record: Structured metrics record.
    """
    if record.sample_count <= 0:
        raise MetricsError("record.sample_count must be positive.")
    if not record.run_id.strip():
        raise MetricsError("record.run_id must not be empty.")
    if not record.split_name.strip():
        raise MetricsError("record.split_name must not be empty.")
    _validate_metric_mapping(record.metrics)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": record.run_id,
        "fold_id": int(record.fold_id),
        "split_name": record.split_name,
        "sample_count": int(record.sample_count),
        "metrics": record.metrics,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

    LOGGER.info(
        "appended_metric_record",
        extra={
            "event": "appended_metric_record",
            "log_path": str(path),
            "run_id": record.run_id,
            "fold_id": int(record.fold_id),
            "split_name": record.split_name,
        },
    )


def _validate_metric_mapping(metrics: dict[str, Any]) -> None:
    if not metrics:
        raise MetricsError("metrics mapping must not be empty.")
    for key, value in metrics.items():
        if not str(key).strip():
            raise MetricsError("metric keys must not be empty.")
        if not isinstance(value, (float, int)):
            raise MetricsError(f"metric '{key}' must be numeric.")


def _stable_metric(value: float | int) -> float:
    return round(float(value), 8)
