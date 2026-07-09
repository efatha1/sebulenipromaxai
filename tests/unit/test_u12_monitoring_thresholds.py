"""Unit tests for U12 monitoring thresholds."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from training.monitoring import MonitoringThresholds, evaluate_live_health


def test_u12_monitoring_emits_alerts_for_latency_advisory_and_stale_report(tmp_path: Path) -> None:
    manifest_path = tmp_path / "active_model_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "latest-report.json"
    report_path.write_text("{}", encoding="utf-8")

    current_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    stale_time = current_time - timedelta(minutes=120)
    report_path.touch()
    import os
    os.utime(report_path, (stale_time.timestamp(), stale_time.timestamp()))

    evaluation = evaluate_live_health(
        latency_ms=(125.0, 130.0, 140.0),
        low_confidence_flags=(True, False, True),
        active_model_manifest_path=manifest_path,
        latest_report_path=report_path,
        thresholds=MonitoringThresholds(
            max_latency_ms=100.0,
            max_low_confidence_rate=0.5,
            max_report_age_minutes=30.0,
        ),
        current_time=current_time,
    )

    assert evaluation.status == "degraded"
    assert {alert.code for alert in evaluation.alerts} == {
        "latency_threshold_breach",
        "advisory_rate_threshold_breach",
        "stale_report_artifact",
    }
