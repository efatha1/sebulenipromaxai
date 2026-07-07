"""Unit tests for U11 report serialization."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from inference.reporting import write_prediction_batch_report
from training.contracts import AnalogRecordContract, PredictionRecordContract, PredictionResponseContract


def test_u11_prediction_report_serialization_uses_prediction_outputs_only(tmp_path: Path) -> None:
    response = PredictionResponseContract(
        prediction=PredictionRecordContract(
            request_id="pred-1",
            reference_ts=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
            horizon=3,
            event_probability=0.75,
            confidence=0.7,
            low_price=98.0,
            high_price=102.0,
            start_estimate=1,
            maturity_estimate=2,
            duration_estimate=2,
            low_confidence_advisory=False,
        ),
        top_k_analogs=(
            AnalogRecordContract(
                analog_id="a1",
                reference_ts=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
                distance=0.1,
                outcome_summary="event_observed=1",
            ),
        ),
        summary_statistics={"analog_count": 1.0, "mean_distance": 0.1},
        grounded_natural_language_explanation="The top 1 training analog grounds this prediction.",
    )

    artifact = write_prediction_batch_report(
        predictions=(response,),
        output_dir=tmp_path,
        report_name="unit-report",
    )

    contents = Path(artifact.output_path).read_text(encoding="utf-8")
    assert artifact.report_type == "prediction_batch"
    assert "prediction_count" in contents
    assert "grounded_natural_language_explanation" in contents
    assert "aggregate_metrics" not in contents
