"""Shared U11 dependencies and service adapters for API and CLI."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from fastapi import Request

from inference.model_store import ActiveModelArtifacts, LoadedActiveModel, load_active_model
from inference.predictor import predict
from inference.reporting import ReportArtifact, write_evaluation_summary, write_prediction_batch_report
from models.explanation import RetrievalMemoryRecord
from training.config_loader import load_config
from training.config_schema import RuntimeConfig
from training.contracts import HealthResponseContract, PredictionRequestContract, PredictionResponseContract
from training.eval_contract import WalkForwardFold
from training.retraining import request_retraining as create_review_gated_retraining_request
from training.trainer import TrainingDataset, run_training

LOGGER = logging.getLogger(__name__)


class ServiceError(ValueError):
    """Raised when shared U11 service operations fail."""


@dataclass(frozen=True)
class TrainingBundle:
    """Serialized training bundle consumed by CLI workflows."""

    dataset: TrainingDataset
    folds: tuple[WalkForwardFold, ...]
    lookbacks_by_timeframe: dict[str, int]
    retrieval_memory: tuple[RetrievalMemoryRecord, ...]


@dataclass(frozen=True)
class ActiveModelManifest:
    """Persisted active-model manifest for API and CLI prediction surfaces."""

    checkpoint_path: str
    lookbacks_by_timeframe: dict[str, int]
    retrieval_memory: tuple[RetrievalMemoryRecord, ...]


@dataclass
class AppServices:
    """Shared service adapters used by both API and CLI."""

    config_path: Path
    active_model_manifest_path: Path | None = None
    current_time: datetime | None = None

    def load_runtime_config(self) -> RuntimeConfig:
        return load_config(self.config_path)

    def health(self) -> HealthResponseContract:
        try:
            model_available = self.active_model_manifest_path is not None and self.active_model_manifest_path.exists()
            return HealthResponseContract(
                status="ok" if model_available else "degraded",
                model_available=model_available,
            )
        except Exception:
            return HealthResponseContract(
                status="degraded",
                model_available=False,
            )

    def load_active_model(self) -> LoadedActiveModel:
        if self.active_model_manifest_path is None:
            raise ServiceError("active_model_manifest_path is required for prediction workflows.")
        manifest = load_active_model_manifest(self.active_model_manifest_path)
        config = self.load_runtime_config()
        return load_active_model(
            config,
            ActiveModelArtifacts(
                checkpoint_path=Path(manifest.checkpoint_path),
                lookbacks_by_timeframe=dict(manifest.lookbacks_by_timeframe),
                retrieval_memory=manifest.retrieval_memory,
            ),
        )

    def inspect_current_model(self) -> dict[str, Any]:
        if self.active_model_manifest_path is None:
            raise ServiceError("active_model_manifest_path is required to inspect the current model.")
        manifest = load_active_model_manifest(self.active_model_manifest_path)
        active_model = self.load_active_model()
        return {
            "model_id": active_model.checkpoint.model_id,
            "checkpoint_path": manifest.checkpoint_path,
            "config_hash": active_model.checkpoint.config_hash,
            "lookbacks_by_timeframe": manifest.lookbacks_by_timeframe,
            "retrieval_memory_size": len(manifest.retrieval_memory),
        }

    def predict_request(self, request: PredictionRequestContract):
        config = self.load_runtime_config()
        active_model = self.load_active_model()
        return predict(request, config, active_model, current_time=self.current_time)

    def generate_prediction_report(
        self,
        prediction_responses: tuple[PredictionResponseContract, ...] | list[PredictionResponseContract],
        *,
        report_name: str | None = None,
    ) -> ReportArtifact:
        config = self.load_runtime_config()
        return write_prediction_batch_report(
            predictions=tuple(prediction_responses),
            output_dir=config.reporting.output_dir,
            report_name=report_name,
        )

    def create_retraining_request(
        self,
        *,
        reason: str,
        candidate_model_id: str | None = None,
    ) -> dict[str, str]:
        if not reason.strip():
            raise ServiceError("reason must not be empty.")
        config = self.load_runtime_config()
        record = create_review_gated_retraining_request(
            output_dir=config.reporting.output_dir,
            instrument_id=config.instrument.instrument_id,
            reason=reason,
            candidate_model_id=candidate_model_id,
            source="manual",
        )
        return {
            "request_id": record.request_id,
            "status": record.status,
            "source": record.source,
            "output_path": str(record.output_path),
        }

    def run_training_bundle(
        self,
        *,
        bundle_path: Path,
        evaluation_output_path: Path,
    ) -> dict[str, str]:
        config = self.load_runtime_config()
        bundle = load_training_bundle(bundle_path)
        summary = run_training(
            config=config,
            folds=bundle.folds,
            dataset=bundle.dataset,
            artifact_root=config.reporting.output_dir,
            checkpoint_metadata={"lookbacks_by_timeframe": dict(bundle.lookbacks_by_timeframe)},
        )
        write_evaluation_summary(summary=summary, output_path=evaluation_output_path)
        result = {
            "evaluation_output_path": str(evaluation_output_path),
            "candidate_checkpoint_path": str(summary.candidate_model.artifact_path),
            "candidate_model_id": summary.candidate_model.model_id,
            "promotion_status": "review_required",
        }
        return result

    def resolve_active_model_manifest_path(self, config: RuntimeConfig) -> Path:
        if self.active_model_manifest_path is not None:
            return self.active_model_manifest_path
        return Path(config.reporting.output_dir) / "active_model_manifest.json"


def get_services(request: Request) -> AppServices:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise ServiceError("application services are not configured.")
    if not isinstance(services, AppServices):
        raise ServiceError("application services must be an AppServices instance.")
    return services


def load_training_bundle(bundle_path: str | Path) -> TrainingBundle:
    path = Path(bundle_path)
    if not path.exists():
        raise ServiceError(f"training bundle does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("dataset", "folds", "lookbacks_by_timeframe", "retrieval_memory")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ServiceError(f"training bundle missing required keys: {missing}")
    return TrainingBundle(
        dataset=payload["dataset"],
        folds=tuple(payload["folds"]),
        lookbacks_by_timeframe=dict(payload["lookbacks_by_timeframe"]),
        retrieval_memory=tuple(payload["retrieval_memory"]),
    )


def load_active_model_manifest(manifest_path: str | Path) -> ActiveModelManifest:
    path = Path(manifest_path)
    if not path.exists():
        raise ServiceError(f"active model manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "checkpoint_path" not in payload or "lookbacks_by_timeframe" not in payload or "retrieval_memory" not in payload:
        raise ServiceError("active model manifest is missing required fields.")
    retrieval_memory = tuple(_deserialize_retrieval_record(item) for item in payload["retrieval_memory"])
    return ActiveModelManifest(
        checkpoint_path=str(payload["checkpoint_path"]),
        lookbacks_by_timeframe={str(key): int(value) for key, value in payload["lookbacks_by_timeframe"].items()},
        retrieval_memory=retrieval_memory,
    )


def save_active_model_manifest(manifest_path: str | Path, manifest: ActiveModelManifest) -> Path:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_path": manifest.checkpoint_path,
        "lookbacks_by_timeframe": manifest.lookbacks_by_timeframe,
        "retrieval_memory": [_serialize_retrieval_record(item) for item in manifest.retrieval_memory],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def serialize_cli_error(exc: Exception) -> dict[str, Any]:
    return {
        "errors": [
            {
                "code": exc.__class__.__name__,
                "message": str(exc),
            }
        ]
    }


def _serialize_retrieval_record(record: RetrievalMemoryRecord) -> dict[str, Any]:
    return {
        "analog_id": record.analog_id,
        "reference_ts": record.reference_ts.isoformat(),
        "latent_vector": list(record.latent_vector),
        "outcome_summary": record.outcome_summary,
        "event_observed": record.event_observed,
        "future_low": record.future_low,
        "future_high": record.future_high,
        "duration_bars": record.duration_bars,
        "source_split": record.source_split,
        "source_fold_id": record.source_fold_id,
    }


def _deserialize_retrieval_record(payload: dict[str, Any]) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        analog_id=str(payload["analog_id"]),
        reference_ts=datetime.fromisoformat(payload["reference_ts"]),
        latent_vector=tuple(float(item) for item in payload["latent_vector"]),
        outcome_summary=str(payload["outcome_summary"]),
        event_observed=float(payload["event_observed"]),
        future_low=float(payload["future_low"]),
        future_high=float(payload["future_high"]),
        duration_bars=float(payload["duration_bars"]) if payload["duration_bars"] is not None else None,
        source_split=str(payload["source_split"]),
        source_fold_id=str(payload["source_fold_id"]),
    )
