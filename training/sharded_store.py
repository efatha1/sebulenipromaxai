"""Disk-backed sharded dataset loader for Kaggle-scale preprocessing outputs.

This module consumes the output produced by the updated `Preprocessing.py`
sharded pipeline (manifest + `windows/`, `targets/`, `reference/`).

Design goals:
- Keep RAM bounded by loading only the required shard slices for each batch.
- Preserve determinism (no stochastic sampling by default).
- Fail early on missing files, shape mismatches, or invalid slice requests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.losses import MultiTaskTargets, MultiTaskTargetsUnified

LOGGER = logging.getLogger(__name__)

MODELED_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d")


class ShardedStoreError(ValueError):
    """Raised when sharded dataset loading fails."""


@dataclass(frozen=True)
class ShardedManifest:
    """Parsed sharded preprocessing manifest."""

    root: Path
    lookbacks_by_timeframe: dict[str, int]
    targets: dict[str, Any]  # Now a dict of manifests per timeframe (Option A)
    windows: dict[str, Any]
    label_selection: dict[str, Any] | None

    @property
    def shard_size(self) -> int:
        # Use 1m shard size as reference (all timeframes should have same shard_size)
        return int(self.targets["1m"]["shard_size"])

    @property
    def num_shards(self) -> int:
        # Use 1m num_shards as reference
        return int(self.targets["1m"]["num_shards"])

    @property
    def total_samples(self) -> int:
        # Use 1m total_samples as reference
        return int(self.targets["1m"]["total_samples"])


def load_sharded_manifest(manifest_path: str | Path) -> ShardedManifest:
    """Load and validate a sharded preprocessing manifest."""
    path = Path(manifest_path)
    if not path.exists():
        raise ShardedStoreError(f"manifest_path does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    layout = payload.get("output_layout") or {}
    root_value = layout.get("root")
    root = Path(root_value) if root_value else path.parent

    lookbacks = payload.get("lookbacks_by_timeframe")
    targets = payload.get("targets")
    windows = payload.get("windows")
    label_selection = payload.get("label_selection")
    if not isinstance(lookbacks, dict) or not lookbacks:
        raise ShardedStoreError("manifest missing lookbacks_by_timeframe.")
    if not isinstance(targets, dict) or not targets:
        raise ShardedStoreError("manifest missing targets.")
    if not isinstance(windows, dict) or not windows:
        raise ShardedStoreError("manifest missing windows.")

    # Validate targets structure (now per-timeframe manifests)
    for tf in MODELED_TIMEFRAMES:
        if tf not in targets:
            raise ShardedStoreError(f"manifest targets missing timeframe={tf}.")
        tf_manifest = targets[tf]
        required_keys = ("total_samples", "shard_size", "num_shards", "column_names")
        missing_keys = [key for key in required_keys if key not in tf_manifest]
        if missing_keys:
            raise ShardedStoreError(f"manifest targets for timeframe={tf} missing required keys: {missing_keys}")

    for tf in MODELED_TIMEFRAMES:
        if tf not in windows:
            raise ShardedStoreError(f"manifest windows missing timeframe={tf}.")

    manifest = ShardedManifest(
        root=root,
        lookbacks_by_timeframe={str(k): int(v) for k, v in lookbacks.items()},
        targets=dict(targets),
        windows=dict(windows),
        label_selection=dict(label_selection) if isinstance(label_selection, dict) else None,
    )
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: ShardedManifest) -> None:
    if set(manifest.lookbacks_by_timeframe.keys()) != set(MODELED_TIMEFRAMES):
        raise ShardedStoreError(f"lookbacks_by_timeframe must contain exactly {MODELED_TIMEFRAMES}.")
    shard_size = manifest.shard_size
    if shard_size <= 0:
        raise ShardedStoreError("shard_size must be positive.")
    num_shards = manifest.num_shards
    if num_shards <= 0:
        raise ShardedStoreError("num_shards must be positive.")
    total = manifest.total_samples
    if total <= 0:
        raise ShardedStoreError("total_samples must be positive.")

    # Enforce a consistent feature_dim across all timeframes (required by the backbone).
    feature_dims = []
    for tf in MODELED_TIMEFRAMES:
        tf_meta = manifest.windows.get(tf, {})
        if "num_features" not in tf_meta:
            raise ShardedStoreError(f"windows metadata missing num_features for timeframe={tf}.")
        feature_dims.append(int(tf_meta["num_features"]))
    if len(set(feature_dims)) != 1:
        raise ShardedStoreError(f"feature_dim mismatch across timeframes: {dict(zip(MODELED_TIMEFRAMES, feature_dims))}")


@dataclass
class _ShardCache:
    shard_id: int | None = None
    windows_by_tf: dict[str, np.ndarray] | None = None
    reference_close: dict[str, np.ndarray] | None = None  # Now per-timeframe
    targets_np: dict[str, dict[str, np.ndarray]] | None = None  # Now per-timeframe


class ShardedDatasetStore:
    """Batch loader over sharded preprocessing outputs."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = load_sharded_manifest(self.manifest_path)
        self._cache = _ShardCache()

    @property
    def root(self) -> Path:
        return self.manifest.root

    @property
    def shard_size(self) -> int:
        return self.manifest.shard_size

    @property
    def num_shards(self) -> int:
        return self.manifest.num_shards

    @property
    def total_samples(self) -> int:
        return self.manifest.total_samples

    @property
    def feature_dim(self) -> int:
        return int(self.manifest.windows["1m"]["num_features"])

    def load_shard(self, shard_id: int) -> None:
        if shard_id < 0 or shard_id >= self.num_shards:
            raise ShardedStoreError(f"Invalid shard_id={shard_id}.")
        if self._cache.shard_id == shard_id:
            return

        windows_dir = self.root / "windows"
        targets_dir = self.root / "targets"
        reference_dir = self.root / "reference"

        windows_by_tf: dict[str, np.ndarray] = {}
        for tf in MODELED_TIMEFRAMES:
            path = windows_dir / tf / f"windows_shard_{shard_id:05d}.npy"
            if not path.exists():
                raise ShardedStoreError(f"Missing windows shard file: {path}")
            windows_by_tf[tf] = np.load(path, allow_pickle=False)

        # Load per-timeframe targets and reference data
        targets_by_tf: dict[str, dict[str, np.ndarray]] = {}
        reference_close_by_tf: dict[str, np.ndarray] = {}
        
        for tf in MODELED_TIMEFRAMES:
            targets_path = targets_dir / f"targets_{tf}_shard_{shard_id:05d}.npz"
            if not targets_path.exists():
                raise ShardedStoreError(f"Missing targets shard file for timeframe={tf}: {targets_path}")
            targets_payload = np.load(targets_path)
            targets_by_tf[tf] = {key: targets_payload[key] for key in targets_payload.files}
            
            close_path = reference_dir / f"reference_close_{tf}_shard_{shard_id:05d}.npy"
            if not close_path.exists():
                raise ShardedStoreError(f"Missing reference_close shard file for timeframe={tf}: {close_path}")
            reference_close_by_tf[tf] = np.load(close_path, allow_pickle=False)

        self._cache = _ShardCache(
            shard_id=shard_id,
            windows_by_tf=windows_by_tf,
            reference_close=reference_close_by_tf,  # Now per-timeframe
            targets_np=targets_by_tf,  # Now per-timeframe
        )

    def get_slice(self, start: int, end: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, MultiTaskTargets]:
        """Load a contiguous slice [start, end) as torch tensors (CPU)."""
        if start < 0 or end < 0 or end < start:
            raise ShardedStoreError(f"Invalid slice: start={start} end={end}")
        if end > self.total_samples:
            raise ShardedStoreError(f"Slice end exceeds dataset size: end={end} total={self.total_samples}")
        if start == end:
            raise ShardedStoreError("Empty slice requested (start == end).")

        windows_parts: dict[str, list[np.ndarray]] = {tf: [] for tf in MODELED_TIMEFRAMES}
        close_parts: list[np.ndarray] = []
        targets_parts: dict[str, list[np.ndarray]] = {
            "event_flag": [],
            "future_low": [],
            "future_high": [],
            "event_start_offset": [],
            "maturity_offset": [],
        }

        cursor = start
        while cursor < end:
            shard_id = cursor // self.shard_size
            local_start = cursor % self.shard_size
            local_end = min(self.shard_size, local_start + (end - cursor))

            self.load_shard(shard_id)
            assert self._cache.windows_by_tf is not None
            assert self._cache.reference_close is not None
            assert self._cache.targets_np is not None

            for tf in MODELED_TIMEFRAMES:
                windows_parts[tf].append(self._cache.windows_by_tf[tf][local_start:local_end])
            # Use 1m reference close for backward compatibility
            close_parts.append(self._cache.reference_close["1m"][local_start:local_end])
            for key in targets_parts:
                if key not in self._cache.targets_np["1m"]:
                    raise ShardedStoreError(f"Targets shard missing key={key} shard_id={shard_id}.")
                targets_parts[key].append(self._cache.targets_np["1m"][key][local_start:local_end])

            cursor += (local_end - local_start)

        windows_by_tf_torch = {
            tf: torch.from_numpy(np.concatenate(windows_parts[tf], axis=0).astype(np.float32, copy=False))
            for tf in MODELED_TIMEFRAMES
        }
        reference_close = torch.from_numpy(np.concatenate(close_parts, axis=0).astype(np.float32, copy=False))
        targets = MultiTaskTargets(
            event_flag=torch.from_numpy(np.concatenate(targets_parts["event_flag"], axis=0).astype(np.float32, copy=False)),
            future_low=torch.from_numpy(np.concatenate(targets_parts["future_low"], axis=0).astype(np.float32, copy=False)),
            future_high=torch.from_numpy(np.concatenate(targets_parts["future_high"], axis=0).astype(np.float32, copy=False)),
            event_start_offset=torch.from_numpy(
                np.concatenate(targets_parts["event_start_offset"], axis=0).astype(np.float32, copy=False)
            ),
            maturity_offset=torch.from_numpy(
                np.concatenate(targets_parts["maturity_offset"], axis=0).astype(np.float32, copy=False)
            ),
            confidence_target=None,
            regime_target=None,
        )
        return windows_by_tf_torch, reference_close, targets

    def get_slice_unified(self, start: int, end: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, MultiTaskTargetsUnified]:
        """Load a contiguous slice [start, end) as unified targets for all 18 combinations."""
        if start < 0 or end < 0 or end < start:
            raise ShardedStoreError(f"Invalid slice: start={start} end={end}")
        if end > self.total_samples:
            raise ShardedStoreError(f"Slice end exceeds dataset size: end={end} total={self.total_samples}")
        if start == end:
            raise ShardedStoreError("Empty slice requested (start == end).")

        windows_parts: dict[str, list[np.ndarray]] = {tf: [] for tf in MODELED_TIMEFRAMES}
        close_parts: list[np.ndarray] = []

        # Initialize unified target parts with nested structure
        unified_targets_parts: dict[str, dict[str, list[np.ndarray]]] = {
            "event_flag": {tf: [] for tf in MODELED_TIMEFRAMES},
            "future_low": {tf: [] for tf in MODELED_TIMEFRAMES},
            "future_high": {tf: [] for tf in MODELED_TIMEFRAMES},
            "event_start_offset": {tf: [] for tf in MODELED_TIMEFRAMES},
            "maturity_offset": {tf: [] for tf in MODELED_TIMEFRAMES},
        }

        cursor = start
        while cursor < end:
            shard_id = cursor // self.shard_size
            local_start = cursor % self.shard_size
            local_end = min(self.shard_size, local_start + (end - cursor))

            self.load_shard(shard_id)
            assert self._cache.windows_by_tf is not None
            assert self._cache.reference_close is not None
            assert self._cache.targets_np is not None

            for tf in MODELED_TIMEFRAMES:
                windows_parts[tf].append(self._cache.windows_by_tf[tf][local_start:local_end])
            # Use 1m reference close for backward compatibility
            close_parts.append(self._cache.reference_close["1m"][local_start:local_end])

            # Parse unified target keys and organize by timeframe
            # New structure: targets_np is per-timeframe dict
            for tf in MODELED_TIMEFRAMES:
                tf_targets = self._cache.targets_np[tf]
                for key in tf_targets:
                    # Key format: {field}_h{horizon}_t{threshold} (timeframe in filename, not key)
                    parts = key.split("_")
                    if len(parts) >= 3:
                        field = parts[0]
                        if field in unified_targets_parts and tf in unified_targets_parts[field]:
                            unified_targets_parts[field][tf].append(
                                tf_targets[key][local_start:local_end]
                            )

            cursor += (local_end - local_start)

        windows_by_tf_torch = {
            tf: torch.from_numpy(np.concatenate(windows_parts[tf], axis=0).astype(np.float32, copy=False))
            for tf in MODELED_TIMEFRAMES
        }
        reference_close = torch.from_numpy(np.concatenate(close_parts, axis=0).astype(np.float32, copy=False))

        # Build unified targets
        event_flag: dict[str, torch.Tensor] = {}
        future_low: dict[str, torch.Tensor] = {}
        future_high: dict[str, torch.Tensor] = {}
        event_start_offset: dict[str, torch.Tensor] = {}
        maturity_offset: dict[str, torch.Tensor] = {}

        for tf in MODELED_TIMEFRAMES:
            event_flag[tf] = torch.from_numpy(
                np.concatenate(unified_targets_parts["event_flag"][tf], axis=0).astype(np.float32, copy=False)
            )
            future_low[tf] = torch.from_numpy(
                np.concatenate(unified_targets_parts["future_low"][tf], axis=0).astype(np.float32, copy=False)
            )
            future_high[tf] = torch.from_numpy(
                np.concatenate(unified_targets_parts["future_high"][tf], axis=0).astype(np.float32, copy=False)
            )
            event_start_offset[tf] = torch.from_numpy(
                np.concatenate(unified_targets_parts["event_start_offset"][tf], axis=0).astype(np.float32, copy=False)
            )
            maturity_offset[tf] = torch.from_numpy(
                np.concatenate(unified_targets_parts["maturity_offset"][tf], axis=0).astype(np.float32, copy=False)
            )

        targets = MultiTaskTargetsUnified(
            event_flag=event_flag,
            future_low=future_low,
            future_high=future_high,
            event_start_offset=event_start_offset,
            maturity_offset=maturity_offset,
        )
        return windows_by_tf_torch, reference_close, targets

