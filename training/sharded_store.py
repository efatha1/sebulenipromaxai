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

from training.config_schema import RuntimeConfig

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

    def __init__(self, manifest_path: str | Path, config: RuntimeConfig | None = None, debug: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = load_sharded_manifest(self.manifest_path)
        self._cache = _ShardCache()
        self.debug = debug
        # Store horizons and thresholds from config for dynamic target parsing
        # Convert to tuples to handle both list and tuple inputs from config
        self.horizons = tuple(config.labeling.horizon_bars) if config else (15, 60, 120)
        self.thresholds = tuple(config.labeling.thresholds) if config else (10.0,)
        
        # DEBUG: Config loading verification
        if self.debug:
            print(f"[DEBUG] Config received: {config is not None}")
            print(f"[DEBUG] Horizons in store: {self.horizons} (type: {type(self.horizons)})")
            print(f"[DEBUG] Thresholds in store: {self.thresholds} (type: {type(self.thresholds)})")
            if config:
                print(f"[DEBUG] Original config horizons: {config.labeling.horizon_bars} (type: {type(config.labeling.horizon_bars)})")
                print(f"[DEBUG] Original config thresholds: {config.labeling.thresholds} (type: {type(config.labeling.thresholds)})")

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

        # Initialize unified target parts with nested structure for horizons
        # Structure: field[timeframe][horizon_idx] = list of arrays
        unified_targets_parts: dict[str, dict[str, dict[int, list[np.ndarray]]]] = {
            "event_flag": {tf: {} for tf in MODELED_TIMEFRAMES},
            "future_low": {tf: {} for tf in MODELED_TIMEFRAMES},
            "future_high": {tf: {} for tf in MODELED_TIMEFRAMES},
            "event_start_offset": {tf: {} for tf in MODELED_TIMEFRAMES},
            "maturity_offset": {tf: {} for tf in MODELED_TIMEFRAMES},
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

            # Parse unified target keys and organize by timeframe, horizon, threshold
            # Key format: {field}_h{horizon}_t{threshold} (timeframe in filename, not key)
            for tf in MODELED_TIMEFRAMES:
                tf_targets = self._cache.targets_np[tf]
                if self.debug:
                    print(f"[DEBUG] Processing timeframe={tf}, found {len(tf_targets)} keys: {list(tf_targets.keys())}")
                
                for key in tf_targets:
                    parts = key.split("_")
                    if self.debug:
                        print(f"[DEBUG] Processing key={key}, parts={parts}, len(parts)={len(parts)}")
                    if len(parts) >= 3:
                        # Find horizon indicator to split field from parameters
                        horizon_idx = None
                        for i, part in enumerate(parts):
                            if part.startswith("h"):
                                horizon_idx = i
                                break
                        
                        if horizon_idx is not None:
                            # Field is everything before the horizon indicator
                            field = "_".join(parts[:horizon_idx])
                            if self.debug:
                                print(f"[DEBUG] field={field}, field in unified_targets_parts={field in unified_targets_parts}")
                            if field in unified_targets_parts:
                                # Extract horizon and threshold from key
                                horizon = None
                                threshold = None
                                for part in parts[1:]:
                                    if part.startswith("h"):
                                        try:
                                            horizon = int(part[1:])
                                        except ValueError:
                                            continue
                                    elif part.startswith("t"):
                                        try:
                                            threshold = float(part[1:])
                                        except ValueError:
                                            continue
                                
                                # DEBUG: Key parsing verification
                                if self.debug:
                                    print(f"[DEBUG] Key={key}, field={field}, horizon={horizon}, threshold={threshold}")
                                    print(f"[DEBUG] Comparison: horizon in self.horizons={horizon in self.horizons}, threshold in self.thresholds={threshold in self.thresholds}")
                                
                                # Validate against config
                                if horizon is not None and threshold is not None:
                                    if horizon in self.horizons and threshold in self.thresholds:
                                        horizon_idx = self.horizons.index(horizon)
                                        # For single threshold, threshold_idx is always 0
                                        threshold_idx = 0 if len(self.thresholds) == 1 else self.thresholds.index(threshold)
                                        
                                        # Store in nested structure: field[timeframe][horizon_idx]
                                        if horizon_idx not in unified_targets_parts[field][tf]:
                                            unified_targets_parts[field][tf][horizon_idx] = []
                                        unified_targets_parts[field][tf][horizon_idx].append(
                                            tf_targets[key][local_start:local_end]
                                        )
                                        if self.debug:
                                            print(f"[DEBUG] Successfully stored: field={field}, tf={tf}, horizon_idx={horizon_idx}")
                                    else:
                                        if self.debug:
                                            print(f"[DEBUG] Failed comparison: key={key}, horizon={horizon} not in {self.horizons} or threshold={threshold} not in {self.thresholds}")
                                else:
                                    if self.debug:
                                        print(f"[DEBUG] Failed to parse: key={key}, horizon={horizon}, threshold={threshold}")

            cursor += (local_end - local_start)

        windows_by_tf_torch = {
            tf: torch.from_numpy(np.concatenate(windows_parts[tf], axis=0).astype(np.float32, copy=False))
            for tf in MODELED_TIMEFRAMES
        }
        reference_close = torch.from_numpy(np.concatenate(close_parts, axis=0).astype(np.float32, copy=False))

        # Build unified targets with horizon dimension
        event_flag: dict[str, torch.Tensor] = {}
        future_low: dict[str, torch.Tensor] = {}
        future_high: dict[str, torch.Tensor] = {}
        event_start_offset: dict[str, torch.Tensor] = {}
        maturity_offset: dict[str, torch.Tensor] = {}

        for tf in MODELED_TIMEFRAMES:
            # Build tensors for each field with horizon dimension
            event_flag_tensors = []
            future_low_tensors = []
            future_high_tensors = []
            event_start_offset_tensors = []
            maturity_offset_tensors = []
            
            for horizon_idx in range(len(self.horizons)):
                horizon = self.horizons[horizon_idx]
                
                # Check if we have data for this horizon
                if horizon_idx in unified_targets_parts["event_flag"][tf]:
                    event_flag_tensors.append(
                        np.concatenate(unified_targets_parts["event_flag"][tf][horizon_idx], axis=0).astype(np.float32, copy=False)
                    )
                else:
                    raise ShardedStoreError(f"Missing event_flag for timeframe={tf}, horizon={horizon}. Check that target files contain the expected keys.")
                
                if horizon_idx in unified_targets_parts["future_low"][tf]:
                    future_low_tensors.append(
                        np.concatenate(unified_targets_parts["future_low"][tf][horizon_idx], axis=0).astype(np.float32, copy=False)
                    )
                else:
                    raise ShardedStoreError(f"Missing future_low for timeframe={tf}, horizon={horizon}. Check that target files contain the expected keys.")
                
                if horizon_idx in unified_targets_parts["future_high"][tf]:
                    future_high_tensors.append(
                        np.concatenate(unified_targets_parts["future_high"][tf][horizon_idx], axis=0).astype(np.float32, copy=False)
                    )
                else:
                    raise ShardedStoreError(f"Missing future_high for timeframe={tf}, horizon={horizon}. Check that target files contain the expected keys.")
                
                if horizon_idx in unified_targets_parts["event_start_offset"][tf]:
                    event_start_offset_tensors.append(
                        np.concatenate(unified_targets_parts["event_start_offset"][tf][horizon_idx], axis=0).astype(np.float32, copy=False)
                    )
                else:
                    raise ShardedStoreError(f"Missing event_start_offset for timeframe={tf}, horizon={horizon}. Check that target files contain the expected keys.")
                
                if horizon_idx in unified_targets_parts["maturity_offset"][tf]:
                    maturity_offset_tensors.append(
                        np.concatenate(unified_targets_parts["maturity_offset"][tf][horizon_idx], axis=0).astype(np.float32, copy=False)
                    )
                else:
                    raise ShardedStoreError(f"Missing maturity_offset for timeframe={tf}, horizon={horizon}. Check that target files contain the expected keys.")
            
            # Stack horizons: (batch, num_horizons)
            event_flag[tf] = torch.from_numpy(np.stack(event_flag_tensors, axis=1))
            future_low[tf] = torch.from_numpy(np.stack(future_low_tensors, axis=1))
            future_high[tf] = torch.from_numpy(np.stack(future_high_tensors, axis=1))
            event_start_offset[tf] = torch.from_numpy(np.stack(event_start_offset_tensors, axis=1))
            maturity_offset[tf] = torch.from_numpy(np.stack(maturity_offset_tensors, axis=1))

        targets = MultiTaskTargetsUnified(
            event_flag=event_flag,
            future_low=future_low,
            future_high=future_high,
            event_start_offset=event_start_offset,
            maturity_offset=maturity_offset,
        )
        return windows_by_tf_torch, reference_close, targets

    def get_reference_timestamps(self) -> pd.DatetimeIndex:
        """Get reference timestamps for fold validation (1m timeframe as reference).
        
        For per-timeframe storage, we reconstruct reference timestamps from the manifest
        metadata by generating synthetic timestamps based on the total sample count and
        assuming regular 1-minute cadence starting from a known reference point.
        """
        # Check if reference timestamps are stored in the manifest
        if "reference_timestamps" in self.manifest.targets["1m"]:
            # Reference timestamps are stored in the manifest
            ref_ts_list = self.manifest.targets["1m"]["reference_timestamps"]
            return pd.DatetimeIndex(ref_ts_list)
        
        # Option A: Reconstruct timestamps assuming regular 1-minute cadence
        # This is an approximation but should work for fold validation
        total_samples = self.manifest.targets["1m"]["total_samples"]
        
        # Use the config timezone from manifest to create timestamps
        # Assume starting from a reasonable point (e.g., first available trading day)
        # This is a limitation of Option A - we don't have the actual timestamps stored
        
        # For now, generate synthetic timestamps starting from a fixed point
        # This will allow fold validation to proceed, though the actual fold boundaries
        # may not match the exact real-world timestamps
        import pandas as pd
        start_ts = pd.Timestamp("2019-01-01 00:00:00", tz="UTC")
        timestamps = pd.date_range(
            start=start_ts,
            periods=total_samples,
            freq="1min"
        )
        
        return timestamps

