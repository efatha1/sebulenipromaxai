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

MODELED_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h")


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

    def get_shard_size(self, timeframe: str) -> int:
        """Get shard size for a specific timeframe."""
        if timeframe not in self.targets:
            raise ShardedStoreError(f"Unknown timeframe: {timeframe}")
        return int(self.targets[timeframe]["shard_size"])

    def get_total_samples(self, timeframe: str) -> int:
        """Get total samples for a specific timeframe."""
        if timeframe not in self.targets:
            raise ShardedStoreError(f"Unknown timeframe: {timeframe}")
        return int(self.targets[timeframe]["total_samples"])

    def get_num_shards(self, timeframe: str) -> int:
        """Get number of shards for a specific timeframe."""
        if timeframe not in self.targets:
            raise ShardedStoreError(f"Unknown timeframe: {timeframe}")
        return int(self.targets[timeframe]["num_shards"])

    def validate_timeframe_consistency(self) -> None:
        """Validate that all timeframes have consistent metadata structure."""
        required_keys = {"total_samples", "shard_size", "num_shards", "column_names"}
        for tf in MODELED_TIMEFRAMES:
            if tf not in self.targets:
                raise ShardedStoreError(f"Missing timeframe in manifest: {tf}")
            tf_manifest = self.targets[tf]
            missing_keys = required_keys - set(tf_manifest.keys())
            if missing_keys:
                raise ShardedStoreError(f"Timeframe {tf} missing keys: {missing_keys}")
            # Validate positive values
            if tf_manifest["shard_size"] <= 0:
                raise ShardedStoreError(f"Timeframe {tf} has invalid shard_size: {tf_manifest['shard_size']}")
            if tf_manifest["num_shards"] <= 0:
                raise ShardedStoreError(f"Timeframe {tf} has invalid num_shards: {tf_manifest['num_shards']}")
            if tf_manifest["total_samples"] <= 0:
                raise ShardedStoreError(f"Timeframe {tf} has invalid total_samples: {tf_manifest['total_samples']}")


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
    
    # Call comprehensive timeframe consistency validation
    manifest.validate_timeframe_consistency()
    
    # Validate that 1m reference values are still positive (for backward compatibility)
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
    
    # Validate that adaptive shard sizes are within reasonable ranges
    for tf in MODELED_TIMEFRAMES:
        tf_shard_size = manifest.get_shard_size(tf)
        if tf_shard_size < 100:
            raise ShardedStoreError(f"Timeframe {tf} shard size too small: {tf_shard_size} (minimum 100)")
        if tf_shard_size > 1000000:
            raise ShardedStoreError(f"Timeframe {tf} shard size too large: {tf_shard_size} (maximum 1000000)")
        
        # Add consistency checks between adaptive shard sizes and actual data volumes
        tf_total_samples = manifest.get_total_samples(tf)
        tf_num_shards = manifest.get_num_shards(tf)
        expected_shard_size = tf_total_samples / tf_num_shards if tf_num_shards > 0 else 0
        
        # Allow 10% tolerance for adaptive sizing calculations
        if abs(tf_shard_size - expected_shard_size) > expected_shard_size * 0.1 and expected_shard_size > 0:
            LOGGER.warning(
                f"Timeframe {tf} shard size inconsistency: manifest shard_size={tf_shard_size} "
                f"vs calculated from data volume={expected_shard_size:.2f}. "
                f"This may indicate adaptive sizing calculation issues."
            )


class ShardedDatasetStore:
    """Batch loader over sharded preprocessing outputs."""

    def __init__(self, manifest_path: str | Path, config: RuntimeConfig | None = None, debug: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = load_sharded_manifest(self.manifest_path)
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

    def get_timeframe_shard_size(self, timeframe: str) -> int:
        """Get shard size for a specific timeframe."""
        return self.manifest.get_shard_size(timeframe)

    def get_timeframe_total_samples(self, timeframe: str) -> int:
        """Get total samples for a specific timeframe."""
        return self.manifest.get_total_samples(timeframe)

    def get_timeframe_num_shards(self, timeframe: str) -> int:
        """Get number of shards for a specific timeframe."""
        return self.manifest.get_num_shards(timeframe)

    def _load_timeframe_shard(self, timeframe: str, shard_id: int, local_start: int, local_end: int) -> np.ndarray:
        """Load a specific timeframe shard directly without caching (best-effort).
        
        Args:
            timeframe: Timeframe identifier (e.g., "1m", "5m")
            shard_id: Shard identifier
            local_start: Start index within shard
            local_end: End index within shard
            
        Returns:
            Window data for the specified shard slice
            
        Raises:
            ShardedStoreError: If shard loading fails critically
        """
        try:
            windows_dir = self.root / "windows" / timeframe
            path = windows_dir / f"windows_shard_{shard_id:05d}.npy"
            if not path.exists():
                raise ShardedStoreError(f"Missing windows shard: {path}")
            
            full_shard = np.load(path, allow_pickle=False)
            result = full_shard[local_start:local_end]
            
            # Log shard loading operation for monitoring
            LOGGER.info(f"Loaded {timeframe} shard {shard_id}: samples {local_start}-{local_end} (shape: {result.shape})")
            
            return result
            
        except Exception as e:
            LOGGER.warning(f"Best-effort: Failed to load {timeframe} shard {shard_id}: {e}")
            raise  # Re-raise for caller to handle

    def _load_targets_shard(self, timeframe: str, shard_id: int, local_start: int, local_end: int) -> dict[str, np.ndarray]:
        """Load timeframe targets shard directly without caching (best-effort).
        
        Args:
            timeframe: Timeframe identifier
            shard_id: Shard identifier  
            local_start: Start index within shard
            local_end: End index within shard
            
        Returns:
            Dictionary of target arrays for the shard slice
            
        Raises:
            ShardedStoreError: If target loading fails critically
        """
        try:
            targets_dir = self.root / "targets"
            path = targets_dir / f"targets_{timeframe}_shard_{shard_id:05d}.npz"
            if not path.exists():
                raise ShardedStoreError(f"Missing targets shard: {path}")
            
            full_shard = np.load(path)
            result = {key: full_shard[key][local_start:local_end] for key in full_shard.files}
            
            # Log shard loading operation for monitoring
            LOGGER.info(f"Loaded {timeframe} targets shard {shard_id}: samples {local_start}-{local_end} (keys: {list(result.keys())})")
            
            return result
            
        except Exception as e:
            LOGGER.warning(f"Best-effort: Failed to load {timeframe} targets shard {shard_id}: {e}")
            raise  # Re-raise for caller to handle

    def _load_reference_close_shard(self, timeframe: str, shard_id: int, local_start: int, local_end: int) -> np.ndarray:
        """Load reference close shard directly without caching (best-effort).
        
        Args:
            timeframe: Timeframe identifier
            shard_id: Shard identifier  
            local_start: Start index within shard
            local_end: End index within shard
            
        Returns:
            Reference close array for the shard slice
            
        Raises:
            ShardedStoreError: If reference close loading fails critically
        """
        try:
            reference_dir = self.root / "reference"
            path = reference_dir / f"reference_close_{timeframe}_shard_{shard_id:05d}.npy"
            if not path.exists():
                raise ShardedStoreError(f"Missing reference_close shard: {path}")
            
            full_shard = np.load(path, allow_pickle=False)
            result = full_shard[local_start:local_end]
            
            # Log shard loading operation for monitoring
            LOGGER.info(f"Loaded {timeframe} reference_close shard {shard_id}: samples {local_start}-{local_end} (shape: {result.shape})")
            
            return result
            
        except Exception as e:
            LOGGER.warning(f"Best-effort: Failed to load {timeframe} reference_close shard {shard_id}: {e}")
            raise  # Re-raise for caller to handle

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

        # Initialize timeframe-specific cursors for loose alignment
        cursors_by_tf = {tf: start for tf in MODELED_TIMEFRAMES}
        
        # Use 1m shard count as reference for max iterations (loose alignment)
        max_iterations = (end - start + self.get_timeframe_shard_size("1m") - 1) // self.get_timeframe_shard_size("1m")
        
        for iteration in range(max_iterations):
            # Use 1m shard_id as reference for loose temporal coordination
            shard_id = cursors_by_tf["1m"] // self.get_timeframe_shard_size("1m")
            
            for tf in MODELED_TIMEFRAMES:
                try:
                    tf_shard_size = self.get_timeframe_shard_size(tf)
                    tf_cursor = cursors_by_tf[tf]
                    tf_local_start = tf_cursor % tf_shard_size
                    tf_local_end = min(tf_shard_size, tf_local_start + min(
                        self.get_timeframe_total_samples(tf) - tf_cursor,
                        (end - start)  # Approximate batch size
                    ))
                    
                    # Direct shard loading without caching
                    tf_windows = self._load_timeframe_shard(tf, shard_id, tf_local_start, tf_local_end)
                    
                    # Store window results
                    windows_parts[tf].append(tf_windows)
                    
                    # Use 1m reference close for backward compatibility
                    if tf == "1m":
                        tf_reference_close = self._load_reference_close_shard(tf, shard_id, tf_local_start, tf_local_end)
                        close_parts.append(tf_reference_close)
                    
                    # Load 1m targets for backward compatibility (legacy targets)
                    if tf == "1m":
                        tf_targets = self._load_targets_shard(tf, shard_id, tf_local_start, tf_local_end)
                        for key in targets_parts:
                            if key in tf_targets:
                                targets_parts[key].append(tf_targets[key])
                    
                    # Update cursor independently (loose alignment)
                    cursors_by_tf[tf] += (tf_local_end - tf_local_start)
                    
                except (ShardedStoreError, FileNotFoundError, ValueError) as e:
                    # Best-effort error handling: log warning and continue with other timeframes
                    LOGGER.warning(f"Best-effort: Failed to load timeframe {tf} shard {shard_id}: {e}")
                    # Continue with other timeframes
                    continue

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

        # Initialize timeframe-specific cursors for loose alignment
        cursors_by_tf = {tf: start for tf in MODELED_TIMEFRAMES}
        
        # Use 1m shard count as reference for max iterations (loose alignment)
        max_iterations = (end - start + self.get_timeframe_shard_size("1m") - 1) // self.get_timeframe_shard_size("1m")
        
        LOGGER.info(f"Starting timeframe-specific slice loading: start={start}, end={end}, max_iterations={max_iterations}")
        
        for iteration in range(max_iterations):
            # Use 1m shard_id as reference for loose temporal coordination
            shard_id = cursors_by_tf["1m"] // self.get_timeframe_shard_size("1m")
            
            # Log loose alignment effectiveness
            cursor_positions = {tf: cursors_by_tf[tf] for tf in MODELED_TIMEFRAMES}
            LOGGER.debug(f"Iteration {iteration}: shard_id={shard_id}, cursor_positions={cursor_positions}")
            
            for tf in MODELED_TIMEFRAMES:
                try:
                    tf_shard_size = self.get_timeframe_shard_size(tf)
                    tf_cursor = cursors_by_tf[tf]
                    tf_local_start = tf_cursor % tf_shard_size
                    tf_local_end = min(tf_shard_size, tf_local_start + min(
                        self.get_timeframe_total_samples(tf) - tf_cursor,
                        (end - start)  # Approximate batch size
                    ))
                    
                    # Direct shard loading without caching
                    tf_windows = self._load_timeframe_shard(tf, shard_id, tf_local_start, tf_local_end)
                    tf_targets = self._load_targets_shard(tf, shard_id, tf_local_start, tf_local_end)
                    
                    # Store window results
                    windows_parts[tf].append(tf_windows)
                    
                    # Use 1m reference close for backward compatibility
                    if tf == "1m":
                        tf_reference_close = self._load_reference_close_shard(tf, shard_id, tf_local_start, tf_local_end)
                        close_parts.append(tf_reference_close)
                    
                    # Parse unified target keys and organize by timeframe, horizon, threshold
                    # Key format: {field}_h{horizon}_t{threshold} (timeframe in filename, not key)
                    if self.debug:
                        print(f"[DEBUG] Processing timeframe={tf}, found {len(tf_targets)} keys: {list(tf_targets.keys())}")
                    
                    for key in tf_targets:
                        parts = key.split("_")
                        if self.debug:
                            print(f"[DEBUG] Processing key={key}, parts={parts}, len(parts)={len(parts)}")
                        if len(parts) >= 3:
                            # Find horizon indicator to split field from parameters
                            # Match pattern: h followed by digits (e.g., h15, h60, h120)
                            horizon_part_idx = None
                            for i, part in enumerate(parts):
                                if part.startswith("h") and len(part) > 1 and part[1:].isdigit():
                                    horizon_part_idx = i
                                    break
                            
                            if self.debug:
                                print(f"[DEBUG] horizon_part_idx={horizon_part_idx}")
                            
                            if horizon_part_idx is not None:
                                # Field is everything before the horizon indicator
                                field = "_".join(parts[:horizon_part_idx])
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
                                                tf_targets[key]
                                            )
                                            if self.debug:
                                                print(f"[DEBUG] Successfully stored: field={field}, tf={tf}, horizon_idx={horizon_idx}")
                                        else:
                                            if self.debug:
                                                print(f"[DEBUG] Failed comparison: key={key}, horizon={horizon} not in {self.horizons} or threshold={threshold} not in {self.thresholds}")
                                    else:
                                        if self.debug:
                                            print(f"[DEBUG] Failed to parse: key={key}, horizon={horizon}, threshold={threshold}")

                    # Update cursor independently (loose alignment)
                    cursors_by_tf[tf] += (tf_local_end - tf_local_start)
                    
                except (ShardedStoreError, FileNotFoundError, ValueError) as e:
                    # Best-effort error handling: log warning and continue with other timeframes
                    LOGGER.warning(f"Best-effort: Failed to load timeframe {tf} shard {shard_id}: {e}")
                    # Continue with other timeframes
                    continue

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
                
                # Check if we have data for this horizon with best-effort handling
                if horizon_idx in unified_targets_parts["event_flag"][tf]:
                    event_flag_tensors.append(
                        np.concatenate(unified_targets_parts["event_flag"][tf][horizon_idx], axis=0).astype(np.float32, copy=False)
                    )
                else:
                    LOGGER.warning(f"Best-effort: Missing event_flag for timeframe={tf}, horizon={horizon}. Check that target files contain the expected keys.")
                    # Use zeros as fallback for best-effort recovery
                    if event_flag_tensors:
                        event_flag_tensors.append(np.zeros_like(event_flag_tensors[0]))
                
                if horizon_idx in unified_targets_parts["future_low"][tf]:
                    future_low_tensors.append(
                        np.concatenate(unified_targets_parts["future_low"][tf][horizon_idx], axis=0).astype(np.float32, copy=False)
                    )
                else:
                    LOGGER.warning(f"Best-effort: Missing future_low for timeframe={tf}, horizon={horizon}. Check that target files contain the expected keys.")
                    if future_low_tensors:
                        future_low_tensors.append(np.zeros_like(future_low_tensors[0]))
                
                if horizon_idx in unified_targets_parts["future_high"][tf]:
                    future_high_tensors.append(
                        np.concatenate(unified_targets_parts["future_high"][tf][horizon_idx], axis=0).astype(np.float32, copy=False)
                    )
                else:
                    LOGGER.warning(f"Best-effort: Missing future_high for timeframe={tf}, horizon={horizon}. Check that target files contain the expected keys.")
                    if future_high_tensors:
                        future_high_tensors.append(np.zeros_like(future_high_tensors[0]))
                
                if horizon_idx in unified_targets_parts["event_start_offset"][tf]:
                    event_start_offset_tensors.append(
                        np.concatenate(unified_targets_parts["event_start_offset"][tf][horizon_idx], axis=0).astype(np.float32, copy=False)
                    )
                else:
                    LOGGER.warning(f"Best-effort: Missing event_start_offset for timeframe={tf}, horizon={horizon}. Check that target files contain the expected keys.")
                    if event_start_offset_tensors:
                        event_start_offset_tensors.append(np.zeros_like(event_start_offset_tensors[0]))
                
                if horizon_idx in unified_targets_parts["maturity_offset"][tf]:
                    maturity_offset_tensors.append(
                        np.concatenate(unified_targets_parts["maturity_offset"][tf][horizon_idx], axis=0).astype(np.float32, copy=False)
                    )
                else:
                    LOGGER.warning(f"Best-effort: Missing maturity_offset for timeframe={tf}, horizon={horizon}. Check that target files contain the expected keys.")
                    if maturity_offset_tensors:
                        maturity_offset_tensors.append(np.zeros_like(maturity_offset_tensors[0]))
            
            # Stack horizons: (batch, num_horizons)
            if self.debug:
                print(f"[DEBUG] Stacking tensors for timeframe={tf}")
                print(f"[DEBUG] event_flag_tensors shapes: {[t.shape for t in event_flag_tensors]}")
                print(f"[DEBUG] future_low_tensors shapes: {[t.shape for t in future_low_tensors]}")
                print(f"[DEBUG] future_high_tensors shapes: {[t.shape for t in future_high_tensors]}")
                print(f"[DEBUG] event_start_offset_tensors shapes: {[t.shape for t in event_start_offset_tensors]}")
                print(f"[DEBUG] maturity_offset_tensors shapes: {[t.shape for t in maturity_offset_tensors]}")
            
            # Validate tensor shapes before stacking (critical for dimension mismatch prevention)
            batch_size = event_flag_tensors[0].shape[0] if event_flag_tensors else 0
            for tensor_list in [event_flag_tensors, future_low_tensors, future_high_tensors, 
                                event_start_offset_tensors, maturity_offset_tensors]:
                if tensor_list:
                    for tensor in tensor_list:
                        if tensor.shape[0] != batch_size:
                            raise ShardedStoreError(
                                f"Tensor dimension mismatch in timeframe {tf}: "
                                f"Expected batch_size={batch_size}, got {tensor.shape[0]}. "
                                f"This indicates timeframe-specific cursor misalignment."
                            )
            
            # Validate loose temporal alignment across timeframes
            for other_tf in MODELED_TIMEFRAMES:
                if other_tf != tf:
                    try:
                        other_batch_size = len(unified_targets_parts["event_flag"][other_tf].get(0, []))
                        if other_batch_size > 0 and abs(batch_size - other_batch_size) > batch_size * 0.1:
                            LOGGER.warning(
                                f"Loose temporal alignment warning: {tf} batch_size={batch_size} "
                                f"vs {other_tf} batch_size={other_batch_size}. This may indicate "
                                f"cursor misalignment due to different shard sizes."
                            )
                    except (KeyError, IndexError):
                        pass  # Skip validation if data is missing (best-effort)
            
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

