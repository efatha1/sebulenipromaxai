"""Unit tests for U12 review guards."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from api.dependencies import ActiveModelManifest, load_active_model_manifest, save_active_model_manifest
from training.evaluate import CandidateModelMetadata, EvaluationSummary, FoldEvaluationSummary
from training.review import approve_candidate


def _build_summary(tmp_path: Path) -> EvaluationSummary:
    candidate_path = tmp_path / "candidate.pt"
    candidate_path.write_text("candidate", encoding="utf-8")
    fold_summary = FoldEvaluationSummary(
        fold_id=0,
        model_id="candidate-model",
        config_hash="cfg123",
        checkpoint_path=candidate_path,
        artifact_dir=tmp_path,
        metrics_log_path=tmp_path / "metrics.jsonl",
        train_metrics={"total_loss": 1.0},
        validation_metrics={"total_loss": 0.4},
        test_metrics={"total_loss": 0.45},
        train_range=(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc)),
        validation_range=(datetime(2026, 1, 3, tzinfo=timezone.utc), datetime(2026, 1, 4, tzinfo=timezone.utc)),
        test_range=(datetime(2026, 1, 5, tzinfo=timezone.utc), datetime(2026, 1, 6, tzinfo=timezone.utc)),
    )
    candidate = CandidateModelMetadata(
        model_id="candidate-model",
        instrument_id="TEST_INSTRUMENT",
        fold_id=0,
        config_hash="cfg123",
        artifact_path=candidate_path,
        metrics={"total_loss": 0.4},
        training_range=(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc)),
    )
    return EvaluationSummary(
        fold_summaries=(fold_summary,),
        aggregate_metrics={"validation_total_loss": 0.4},
        candidate_model=candidate,
    )


def test_u12_rejection_does_not_update_active_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "active_model_manifest.json"
    save_active_model_manifest(
        manifest_path,
        ActiveModelManifest(
            checkpoint_path=str(tmp_path / "current.pt"),
            lookbacks_by_timeframe={"1m": 1, "5m": 1, "15m": 1, "1h": 1, "4h": 1, "1d": 1},
            retrieval_memory=tuple(),
        ),
    )
    summary = _build_summary(tmp_path)

    decision = approve_candidate(
        candidate_summary=summary,
        active_model_manifest_path=manifest_path,
        lookbacks_by_timeframe={"1m": 1, "5m": 1, "15m": 1, "1h": 1, "4h": 1, "1d": 1},
        retrieval_memory=tuple(),
        output_dir=tmp_path,
        approved=False,
        review_reason="metrics require manual hold",
        reviewer_id="reviewer-1",
        reviewed_at=datetime(2026, 1, 7, tzinfo=timezone.utc),
    )

    manifest = load_active_model_manifest(manifest_path)
    assert decision.status == "rejected"
    assert decision.rollback_manifest_path is None
    assert manifest.checkpoint_path == str(tmp_path / "current.pt")
