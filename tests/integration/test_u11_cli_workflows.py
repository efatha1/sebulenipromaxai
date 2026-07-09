"""Integration tests for U11 CLI workflows."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import torch
from typer.testing import CliRunner

from api.dependencies import AppServices, TrainingBundle, load_training_bundle
from models.common import TIMEFRAMES
from models.explanation import RetrievalMemoryRecord
from models.losses import MultiTaskTargets
from training.cli import app
from training.config_loader import load_config
from training.folds import build_walk_forward_folds
from training.review import approve_candidate
from training.trainer import TrainingDataset, run_training


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
instrument:
  instrument_id: TEST_INSTRUMENT
data_source:
  ohlc_path: data/test.csv
time:
  source_timezone: UTC
  runtime_timezone: UTC
  daily_close:
    time: "17:00"
    timezone: America/New_York
sessions:
  calendar_name: default
  weekend_policy: include
  holiday_policy: include
  definitions:
    - name: primary
      start: "00:00"
      end: "23:59"
resampling:
  base_timeframe: 1m
  target_timeframes: ["5m", "15m", "1h", "4h", "1d"]
features:
  enabled_features: ["returns", "calendar_time"]
  deterministic_derived_features: ["returns", "calendar_time"]
labeling:
  thresholds: [10.0]
  horizon_bars: [3, 5]
walk_forward:
  train_bars: 4
  validation_bars: 2
  test_bars: 2
  step_bars: 2
training:
  random_seed: 7
  batch_size: 2
  learning_rate: 0.001
  max_epochs: 1
  device_preference: cpu
  allow_nondeterministic: false
retrieval:
  top_k_analogs: 2
api:
  host: 127.0.0.1
  port: 8000
reporting:
  output_dir: "%s"
schedules:
  reporting_cron: "0 9 * * *"
  retraining_review_cron: "0 0 * * 1"
"""
        % str(tmp_path).replace("\\", "/"),
        encoding="utf-8",
    )
    return path


def _make_bundle(tmp_path: Path, bundle_path: Path, config_path: Path) -> None:
    config = load_config(config_path)
    reference_ts = [datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc) for i in range(22)]
    labeled_df = pd.DataFrame({"reference_ts": reference_ts, "ambiguous": [False] * len(reference_ts)})
    folds = build_walk_forward_folds(labeled_df, config)
    torch.manual_seed(5)
    feature_dim = 7
    windows_by_timeframe = {timeframe: torch.randn(len(reference_ts), 3, feature_dim) for timeframe in TIMEFRAMES}
    dataset = TrainingDataset(
        reference_ts=tuple(reference_ts),
        windows_by_timeframe=windows_by_timeframe,
        reference_close=torch.linspace(100.0, 101.1, steps=len(reference_ts)),
        targets=MultiTaskTargets(
            event_flag=torch.tensor([0.0, 1.0] * 11),
            future_low=torch.linspace(98.0, 99.1, steps=len(reference_ts)),
            future_high=torch.linspace(101.0, 103.2, steps=len(reference_ts)),
            event_start_offset=torch.tensor([-1.0, 2.0] * 11),
            maturity_offset=torch.tensor([-1.0, 4.0] * 11),
            confidence_target=torch.tensor([0.1, 0.9] * 11),
            regime_target=None,
        ),
    )
    bundle = TrainingBundle(
        dataset=dataset,
        folds=tuple(folds),
        lookbacks_by_timeframe={timeframe: 1 for timeframe in TIMEFRAMES},
        retrieval_memory=tuple(
            RetrievalMemoryRecord(
                analog_id=f"fold-0:2026-01-01T00:0{index}:00+00:00",
                reference_ts=datetime(2026, 1, 1, 0, index, tzinfo=timezone.utc),
                latent_vector=tuple(0.0 for _ in range(feature_dim)),
                outcome_summary="event_observed=1; future_low=98.0000; future_high=102.0000; duration_bars=2.0",
                event_observed=1.0,
                future_low=98.0,
                future_high=102.0,
                duration_bars=2.0,
                source_split="train",
                source_fold_id="fold-0",
            )
            for index in range(1, 3)
        ),
    )
    torch.save(
        {
            "dataset": bundle.dataset,
            "folds": bundle.folds,
            "lookbacks_by_timeframe": bundle.lookbacks_by_timeframe,
            "retrieval_memory": bundle.retrieval_memory,
        },
        bundle_path,
    )


def test_u11_cli_training_prediction_and_reporting_flow(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = _write_config(tmp_path)
    bundle_path = tmp_path / "bundle.pt"
    manifest_path = tmp_path / "active_model_manifest.json"
    evaluation_path = tmp_path / "evaluation.json"
    request_path = tmp_path / "request.json"
    responses_path = tmp_path / "responses.json"
    _make_bundle(tmp_path, bundle_path, config_path)

    request_payload = {
        "instrument_id": "TEST_INSTRUMENT",
        "bars_1m": [
            {
                "timestamp": (datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=index)).isoformat(),
                "open": 100.0 + index,
                "high": 100.5 + index,
                "low": 99.5 + index,
                "close": 100.25 + index,
            }
            for index in range(6)
        ],
        "horizon_mode": "single",
        "horizon_bars": 3,
        "threshold": 10.0,
        "top_k_analogs": 1,
    }
    request_path.write_text(json.dumps(request_payload), encoding="utf-8")

    train = runner.invoke(
        app,
        [
            "train",
            "--config-path",
            str(config_path),
            "--bundle-path",
            str(bundle_path),
            "--evaluation-output-path",
            str(evaluation_path),
        ],
    )
    evaluate = runner.invoke(
        app,
        [
            "evaluate",
            "--config-path",
            str(config_path),
            "--bundle-path",
            str(bundle_path),
            "--evaluation-output-path",
            str(tmp_path / "evaluation_only.json"),
        ],
    )
    assert train.exit_code == 0
    assert evaluate.exit_code == 0

    train_payload = json.loads(train.stdout)
    assert train_payload["promotion_status"] == "review_required"
    assert not manifest_path.exists()
    assert evaluation_path.exists()

    services = AppServices(config_path=config_path, active_model_manifest_path=manifest_path)
    config = services.load_runtime_config()
    bundle = load_training_bundle(bundle_path)
    summary = run_training(
        config=config,
        folds=bundle.folds,
        dataset=bundle.dataset,
        artifact_root=config.reporting.output_dir,
        checkpoint_metadata={"lookbacks_by_timeframe": dict(bundle.lookbacks_by_timeframe)},
    )
    approve_candidate(
        candidate_summary=summary,
        active_model_manifest_path=manifest_path,
        lookbacks_by_timeframe=bundle.lookbacks_by_timeframe,
        retrieval_memory=bundle.retrieval_memory,
        output_dir=config.reporting.output_dir,
        approved=True,
        review_reason="cli integration approval",
        reviewer_id="integration-test",
        reviewed_at=datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc),
    )

    predict_result = runner.invoke(
        app,
        [
            "predict",
            "--config-path",
            str(config_path),
            "--active-model-manifest-path",
            str(manifest_path),
            "--request-path",
            str(request_path),
        ],
    )
    assert predict_result.exit_code == 0

    prediction_payload = json.loads(predict_result.stdout)
    responses_path.write_text(json.dumps(prediction_payload), encoding="utf-8")

    inspect_result = runner.invoke(
        app,
        [
            "inspect-model",
            "--config-path",
            str(config_path),
            "--active-model-manifest-path",
            str(manifest_path),
        ],
    )
    report = runner.invoke(
        app,
        [
            "generate-report",
            "--config-path",
            str(config_path),
            "--responses-path",
            str(responses_path),
            "--report-name",
            "cli-report",
        ],
    )

    assert inspect_result.exit_code == 0
    assert report.exit_code == 0
    assert manifest_path.exists()
    assert json.loads(predict_result.stdout)["prediction"]["horizon"] == 3
    assert json.loads(inspect_result.stdout)["model_id"] == summary.candidate_model.model_id
    assert Path(json.loads(report.stdout)["output_path"]).exists()
