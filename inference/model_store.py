"""U10 active model loading from U9 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inference.retrieval import build_retrieval_index
from models.common import TIMEFRAMES
from models.explanation import RetrievalIndex, RetrievalMemoryRecord
from training.checkpointing import CheckpointError, LoadedCheckpoint, load_checkpoint, stable_config_hash
from training.config_schema import RuntimeConfig
from training.trainer import TrainingModel


class ModelStoreError(ValueError):
    """Raised when loading an active model fails."""


@dataclass(frozen=True)
class ActiveModelArtifacts:
    """Typed artifact bundle required for local inference."""

    checkpoint_path: Path
    lookbacks_by_timeframe: dict[str, int]
    retrieval_memory: tuple[RetrievalMemoryRecord, ...]


@dataclass(frozen=True)
class LoadedActiveModel:
    """Loaded inference model plus retrieval support."""

    checkpoint: LoadedCheckpoint
    model: TrainingModel
    lookbacks_by_timeframe: dict[str, int]
    retrieval_index: RetrievalIndex


def load_active_model(
    config: RuntimeConfig,
    artifacts: ActiveModelArtifacts,
) -> LoadedActiveModel:
    """Load the active checkpoint and retrieval memory for inference."""
    _validate_lookbacks(artifacts.lookbacks_by_timeframe)
    checkpoint = load_checkpoint(artifacts.checkpoint_path)
    expected_hash = stable_config_hash(config)
    if checkpoint.config_hash != expected_hash:
        raise ModelStoreError("checkpoint config_hash does not match the active runtime configuration.")
    checkpoint_lookbacks = checkpoint.metadata.get("lookbacks_by_timeframe")
    if checkpoint_lookbacks is None:
        raise ModelStoreError("checkpoint metadata missing lookbacks_by_timeframe.")
    normalized_checkpoint_lookbacks = {str(key): int(value) for key, value in checkpoint_lookbacks.items()}
    if normalized_checkpoint_lookbacks != artifacts.lookbacks_by_timeframe:
        raise ModelStoreError("active-model lookbacks do not match the checkpoint training metadata.")

    try:
        feature_dim = _infer_feature_dim(checkpoint)
    except CheckpointError as exc:
        raise ModelStoreError(str(exc)) from exc

    model = TrainingModel(
        config=config,
        feature_dim=feature_dim,
        max_horizon_bars=int(max(config.labeling.horizon_bars)),
    )
    model.load_state_dict(checkpoint.state_dict)
    model.eval()

    retrieval_index = build_retrieval_index(artifacts.retrieval_memory)
    return LoadedActiveModel(
        checkpoint=checkpoint,
        model=model,
        lookbacks_by_timeframe=dict(artifacts.lookbacks_by_timeframe),
        retrieval_index=retrieval_index,
    )


def _infer_feature_dim(checkpoint: LoadedCheckpoint) -> int:
    weight_key = "backbone.encoders.1m._proj.weight"
    if weight_key not in checkpoint.state_dict:
        raise CheckpointError(f"checkpoint state_dict missing required key: {weight_key}")
    weight = checkpoint.state_dict[weight_key]
    if not hasattr(weight, "shape") or len(weight.shape) != 2:
        raise CheckpointError("checkpoint feature projection weight must have rank 2.")
    return int(weight.shape[1])


def _validate_lookbacks(lookbacks_by_timeframe: dict[str, int]) -> None:
    if set(lookbacks_by_timeframe.keys()) != set(TIMEFRAMES):
        raise ModelStoreError(f"lookbacks_by_timeframe must contain exactly {TIMEFRAMES}.")
    for timeframe, lookback in lookbacks_by_timeframe.items():
        if lookback <= 0:
            raise ModelStoreError(f"lookback must be positive for timeframe={timeframe}.")
