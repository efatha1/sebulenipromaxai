"""U9 reproducible checkpointing helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from training.config_schema import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class CheckpointError(ValueError):
    """Raised when checkpoint save/load fails."""


@dataclass(frozen=True)
class CheckpointArtifact:
    """Metadata for a saved training checkpoint."""

    model_id: str
    config_hash: str
    fold_id: int
    checkpoint_path: Path
    artifact_dir: Path


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Loaded checkpoint payload and metadata."""

    model_id: str
    config_hash: str
    fold_id: int
    metrics: dict[str, float]
    metadata: dict[str, Any]
    state_dict: dict[str, object]
    optimizer_state_dict: dict[str, object] | None


def stable_config_hash(config: RuntimeConfig) -> str:
    """Build a deterministic hash for the validated runtime configuration."""
    payload = config.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def save_checkpoint(
    *,
    artifact_root: str | Path,
    config: RuntimeConfig,
    fold_id: int,
    model: nn.Module,
    optimizer: Optimizer | None,
    metrics: dict[str, float],
    metadata: dict[str, Any] | None = None,
) -> CheckpointArtifact:
    """Save a reproducible checkpoint.

    Args:
        artifact_root: Base artifact directory.
        config: Validated runtime configuration.
        fold_id: Fold identifier.
        model: Training model module.
        optimizer: Optional optimizer.
        metrics: Validation metrics associated with the checkpoint.

    Returns:
        Saved checkpoint artifact metadata.
    """
    if fold_id < 0:
        raise CheckpointError("fold_id must be non-negative.")
    if not metrics:
        raise CheckpointError("metrics must not be empty.")

    config_hash = stable_config_hash(config)
    model_id = f"{config.instrument.instrument_id}-{config_hash}-fold-{fold_id}"
    artifact_dir = Path(artifact_root) / "training_runs" / config_hash / f"fold_{fold_id}"
    checkpoint_path = artifact_dir / "model.pt"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_id": model_id,
        "config_hash": config_hash,
        "fold_id": int(fold_id),
        "instrument_id": config.instrument.instrument_id,
        "metrics": dict(metrics),
        "metadata": dict(metadata or {}),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(payload, checkpoint_path)

    LOGGER.info(
        "saved_checkpoint",
        extra={
            "event": "saved_checkpoint",
            "model_id": model_id,
            "fold_id": int(fold_id),
            "checkpoint_path": str(checkpoint_path),
            "config_hash": config_hash,
        },
    )
    return CheckpointArtifact(
        model_id=model_id,
        config_hash=config_hash,
        fold_id=int(fold_id),
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
    )


def load_checkpoint(checkpoint_path: str | Path) -> LoadedCheckpoint:
    """Load a checkpoint payload.

    Args:
        checkpoint_path: Path to the saved checkpoint.

    Returns:
        Loaded checkpoint payload.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise CheckpointError(f"checkpoint_path does not exist: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = ("model_id", "config_hash", "fold_id", "metrics", "metadata", "state_dict", "optimizer_state_dict")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise CheckpointError(f"checkpoint payload missing required keys: {missing}")

    metrics = payload["metrics"]
    if not isinstance(metrics, dict) or not metrics:
        raise CheckpointError("checkpoint metrics must be a non-empty mapping.")

    LOGGER.info(
        "loaded_checkpoint",
        extra={
            "event": "loaded_checkpoint",
            "model_id": payload["model_id"],
            "fold_id": int(payload["fold_id"]),
            "checkpoint_path": str(path),
            "config_hash": payload["config_hash"],
        },
    )
    return LoadedCheckpoint(
        model_id=str(payload["model_id"]),
        config_hash=str(payload["config_hash"]),
        fold_id=int(payload["fold_id"]),
        metrics=dict(metrics),
        metadata=dict(payload["metadata"]),
        state_dict=dict(payload["state_dict"]),
        optimizer_state_dict=payload["optimizer_state_dict"],
    )
