```
"""Preprocessing pipeline with nested micro-checkpointing - FIXED v3"""
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
from training.config_loader import load_config
from training.data_loader import load_ohlc_frame
from training.calendar import normalize_calendar
from training.data_quality import validate_bar_sequence
from training.resample import resample_timeframes
from training.features import build_features
from training.labeling import generate_labels
from training.folds import build_walk_forward_folds
from training.windowing import build_windows
from models.common import TIMEFRAMES
from models.losses import MultiTaskTargets
from api.dependencies import TrainingBundle
import pandas as pd
import numpy as np
import pickle


class MicroCheckpoint:
    """Fine-grained checkpointing for long-running operations."""
    
    def __init__(self, process_name: str, checkpoint_dir: str = "artifacts/checkpoints"):
        self.process_name = process_name
        self.checkpoint_dir = Path(checkpoint_dir) / process_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.checkpoint_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create new."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {
            "process_name": self.process_name,
            "created_at": datetime.now().isoformat(),
            "checkpoints": {},
            "last_updated": None,
        }
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save(self, checkpoint_id: str, data: Any, meta: Dict[str, Any] = None) -> Path:
        """Save a micro-checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        # Always use pickle for safe serialization
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Update metadata
        self.metadata["checkpoints"][checkpoint_id] = {
            "saved_at": datetime.now().isoformat(),
            "file_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
            **(meta or {}),
        }
        self._save_metadata()
        
        return checkpoint_path
    
    def load(self, checkpoint_id: str) -> Optional[Any]:
        """Load a micro-checkpoint if it exists."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            print(f"  ✓ Loaded checkpoint: {checkpoint_id}")
            return data
        except Exception as e:
            print(f"  ✗ Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def exists(self, checkpoint_id: str) -> bool:
        """Check if a checkpoint exists."""
        return (self.checkpoint_dir / f"{checkpoint_id}.pkl").exists()
    
    def get_latest(self) -> Optional[str]:
        """Get the ID of the most recent checkpoint."""
        if not self.metadata["checkpoints"]:
            return None
        return sorted(self.metadata["checkpoints"].keys())[-1]
    
    def list_checkpoints(self) -> List[str]:
        """List all checkpoints for this process."""
        return sorted(self.metadata["checkpoints"].keys())
    
    def clear(self) -> None:
        """Clear all checkpoints for this process."""
        for checkpoint_file in self.checkpoint_dir.glob("*.pkl"):
            checkpoint_file.unlink()
        self.metadata["checkpoints"] = {}
        self._save_metadata()
        print(f"  Cleared all checkpoints for {self.process_name}")


class PreprocessingCheckpoint:
    """Manages high-level stage checkpoints."""
    
    STAGES = [
        "loaded_ohlc",
        "normalized_calendar",
        "validated_bars",
        "resampled_timeframes",
        "built_features",
        "generated_labels",
        "created_folds",
        "built_windows",
        "aligned_labels",
        "created_dataset",
        "saved_bundle",
    ]
    
    def __init__(self, checkpoint_dir: str = "artifacts/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.checkpoint_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create new."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {
            "created_at": datetime.now().isoformat(),
            "completed_stages": [],
            "stage_timings": {},
        }
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_checkpoint(self, stage: str, data: Any, timing: float = 0.0) -> Path:
        """Save a checkpoint for a specific stage."""
        if stage not in self.STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        
        checkpoint_path = self.checkpoint_dir / f"{stage}.pkl"
        
        # Always use pickle for safe serialization
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"✗ Failed to save checkpoint {stage}: {e}")
            return None
        
        if stage not in self.metadata["completed_stages"]:
            self.metadata["completed_stages"].append(stage)
        self.metadata["stage_timings"][stage] = timing
        self.metadata["last_updated"] = datetime.now().isoformat()
        self._save_metadata()
        
        file_size = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"✓ Stage checkpoint: {stage} ({timing:.2f}s, {file_size:.2f}MB)")
        return checkpoint_path
    
    def load_checkpoint(self, stage: str) -> Optional[Any]:
        """Load a checkpoint if it exists."""
        checkpoint_path = self.checkpoint_dir / f"{stage}.pkl"
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            if data is None:
                return None
            print(f"  ✓ Loaded stage: {stage}")
            return data
        except Exception as e:
            print(f"  ✗ Failed to load stage {stage}: {e}")
            return None
    
    def stage_completed(self, stage: str) -> bool:
        """Check if a stage has been completed."""
        return stage in self.metadata["completed_stages"]
    
    def reset_from_stage(self, stage: str) -> None:
        """Reset checkpoints from a specific stage onwards."""
        stage_idx = self.STAGES.index(stage)
        stages_to_remove = self.STAGES[stage_idx:]
        
        for s in stages_to_remove:
            checkpoint_path = self.checkpoint_dir / f"{s}.pkl"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"Removed checkpoint: {s}")
        
        self.metadata["completed_stages"] = [
            s for s in self.metadata["completed_stages"] 
            if self.STAGES.index(s) < stage_idx
        ]
        self._save_metadata()
        print(f"Reset checkpoints from stage: {stage}")
    
    def get_status(self) -> str:
        """Get a human-readable status."""
        status = "Pipeline Status:\n"
        for stage in self.STAGES:
            completed = "✓" if stage in self.metadata["completed_stages"] else "✗"
            timing = self.metadata["stage_timings"].get(stage, 0.0)
            checkpoint_path = self.checkpoint_dir / f"{stage}.pkl"
            exists = "📁" if checkpoint_path.exists() else "  "
            status += f"  {completed} {stage} ({timing:.2f}s) {exists}\n"
        return status


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def _load_or_none(checkpoint, stage: str) -> Optional[Any]:
    """Load checkpoint safely, return None if not found or failed."""
    data = checkpoint.load_checkpoint(stage)
    return data


# ============================================================================
# LONG-RUNNING PROCESS WRAPPERS WITH MICRO-CHECKPOINTING
# ============================================================================

def generate_labels_with_checkpoints(bars_1m, config, checkpoint_interval: int = 1):
    """
    Generate labels with micro-checkpoints at regular intervals.
    """
    if bars_1m is None:
        raise ValueError("bars_1m is None - previous checkpoint loading failed")
    
    micro_cp = MicroCheckpoint("generate_labels")
    existing = micro_cp.load("final_labels")
    
    if existing is not None:
        print(f"✓ Loaded cached labels from micro-checkpoint")
        return existing
    
    print("Generating labels with micro-checkpoints...")
    start_time = time.time()
    
    # Generate all labels normally
    labels = generate_labels(bars_1m, config, horizon_mode="multi")
    
    # Save incrementally by horizon-threshold pair
    total_combinations = len(labels)
    for idx, (key, df) in enumerate(labels.items(), 1):
        horizon, threshold = key
        checkpoint_id = f"h{horizon}_t{threshold}"
        micro_cp.save(checkpoint_id, df, meta={
            "horizon": int(horizon),
            "threshold": float(threshold),
            "rows": len(df),
        })
        
        if idx % checkpoint_interval == 0 or idx == total_combinations:
            elapsed = time.time() - start_time
            print(f"  ✓ Checkpointed {idx}/{total_combinations} label sets ({elapsed:.1f}s)")
    
    # Save final combined result
    micro_cp.save("final_labels", labels, meta={
        "total_combinations": len(labels),
        "total_samples": sum(len(df) for df in labels.values()),
    })
    
    print(f"✓ Label generation complete ({time.time() - start_time:.1f}s)")
    return labels


def build_windows_with_checkpoints(features_by_tf, lookbacks, checkpoint_interval: int = 2):
    """
    Build windows with micro-checkpoints per timeframe.
    """
    micro_cp = MicroCheckpoint("build_windows")
    existing = micro_cp.load("final_windows")
    
    if existing is not None:
        print(f"✓ Loaded cached windows from micro-checkpoint")
        return existing
    
    print("Building windows with micro-checkpoints...")
    start_time = time.time()
    
    from training.windowing import _build_timeframe_windows
    
    windows = {}
    timeframes = list(features_by_tf.keys())
    
    for idx, timeframe in enumerate(timeframes, 1):
        tf_start = time.time()
        windows[timeframe] = _build_timeframe_windows(
            features_by_tf[timeframe],
            timeframe=timeframe,
            lookback=lookbacks[timeframe]
        )
        tf_time = time.time() - tf_start
        
        # Save checkpoint for this timeframe
        micro_cp.save(f"windows_{timeframe}", windows[timeframe], meta={
            "timeframe": timeframe,
            "lookback": lookbacks[timeframe],
            "reference_ts_count": len(windows[timeframe].reference_ts),
            "window_shape": list(windows[timeframe].windows.shape),
        })
        
        if idx % checkpoint_interval == 0 or idx == len(timeframes):
            elapsed = time.time() - start_time
            print(f"  ✓ Built {idx}/{len(timeframes)} timeframes ({elapsed:.1f}s, last TF: {tf_time:.1f}s)")
    
    # Save final combined result
    micro_cp.save("final_windows", windows, meta={
        "timeframes": list(windows.keys()),
        "total_reference_ts": len(windows["1m"].reference_ts),
    })
    
    print(f"✓ Window building complete ({time.time() - start_time:.1f}s)")
    return windows


def create_dataset_with_checkpoints(
    windows, label_df_aligned, bars_1m, batch_size: int = 10000
):
    """
    Create dataset with micro-checkpoints for large arrays.
    """
    micro_cp = MicroCheckpoint("create_dataset")
    existing = micro_cp.load("final_dataset")
    
    if existing is not None:
        print(f"✓ Loaded cached dataset from micro-checkpoint")
        return existing
    
    print(f"Creating dataset with batch checkpoints (batch_size={batch_size})...")
    start_time = time.time()
    
    window_ts_list = list(windows["1m"].reference_ts)
    aligned_indices = [window_ts_list.index(ts) for ts in label_df_aligned["reference_ts"]]
    
    total_samples = len(label_df_aligned)
    n_batches = (total_samples + batch_size - 1) // batch_size
    
    # Process in batches
    all_windows_filtered = {tf: [] for tf in windows.keys()}
    all_reference_close = []
    all_targets = {
        "event_flag": [],
        "future_low": [],
        "future_high": [],
        "event_start_offset": [],
        "maturity_offset": [],
        "confidence_target": [],
    }
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_samples)
        batch_indices = aligned_indices[batch_start:batch_end]
        
        # Slice windows for this batch
        for tf in windows.keys():
            batch_windows = windows[tf].windows[batch_indices]
            all_windows_filtered[tf].append(batch_windows)
        
        # Slice reference close
        batch_reference_close = torch.tensor(
            [bars_1m.loc[ts, "close"] 
             for ts in label_df_aligned["reference_ts"].iloc[batch_start:batch_end]],
            dtype=torch.float32
        )
        all_reference_close.append(batch_reference_close)
        
        # Slice targets
        for key in all_targets:
            batch_targets = torch.tensor(
                label_df_aligned[key].iloc[batch_start:batch_end].values,
                dtype=torch.float32
            )
            all_targets[key].append(batch_targets)
        
        # Save batch checkpoint
        micro_cp.save(f"batch_{batch_idx:05d}", {
            "windows": {tf: all_windows_filtered[tf][-1] for tf in windows.keys()},
            "reference_close": batch_reference_close,
            "targets": {k: all_targets[k][-1] for k in all_targets},
        }, meta={
            "batch_idx": batch_idx,
            "batch_size": batch_end - batch_start,
            "total_batches": n_batches,
        })
        
        if (batch_idx + 1) % max(1, n_batches // 5) == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - start_time
            progress = ((batch_idx + 1) / n_batches) * 100
            print(f"  ✓ Processed {batch_idx + 1}/{n_batches} batches ({progress:.1f}%, {elapsed:.1f}s)")
    
    # Concatenate all batches
    windows_filtered = {
        tf: np.concatenate(all_windows_filtered[tf], axis=0)
        for tf in windows.keys()
    }
    reference_close = torch.cat(all_reference_close, dim=0)
    targets_final = {
        k: torch.cat(all_targets[k], dim=0)
        for k in all_targets
    }
    
    # Create final dataset
    dataset = TrainingDataset(
        reference_ts=tuple(label_df_aligned["reference_ts"]),
        windows_by_timeframe={
            tf: torch.tensor(windows_filtered[tf], dtype=torch.float32)
            for tf in windows.keys()
        },
        reference_close=reference_close,
        targets=MultiTaskTargets(
            event_flag=targets_final["event_flag"],
            future_low=targets_final["future_low"],
            future_high=targets_final["future_high"],
            event_start_offset=targets_final["event_start_offset"],
            maturity_offset=targets_final["maturity_offset"],
            confidence_target=targets_final["confidence_target"],
            regime_target=None,
        ),
    )
    
    # Save final dataset
    micro_cp.save("final_dataset", dataset, meta={
        "total_samples": len(label_df_aligned),
        "batch_size": batch_size,
        "total_batches": n_batches,
    })
    
    print(f"✓ Dataset creation complete ({time.time() - start_time:.1f}s)")
    return dataset


# ============================================================================
# MAIN PIPELINE
# ============================================================================

# Initialize checkpointing
checkpoint = PreprocessingCheckpoint()
print(checkpoint.get_status())
print()

# 1. Load config
print("Loading configuration...")
config = load_config("config/base.yaml")

# 2. Load OHLC data
stage = "loaded_ohlc"
print(f"\nStage: {stage}")
if checkpoint.stage_completed(stage):
    bars_1m = _load_or_none(checkpoint, stage)
else:
    bars_1m = None

if bars_1m is None:
    print("Loading OHLC data...")
    start = time.time()
    bars_1m = load_ohlc_frame(config)
    print(f"Loaded {len(bars_1m)} 1-minute bars")
    before = len(bars_1m)
    bars_1m = bars_1m[~bars_1m.index.duplicated(keep='first')]
    print(f"Removed {before - len(bars_1m)} duplicates")
    checkpoint.save_checkpoint(stage, bars_1m, timing=time.time() - start)

# 3. Normalize calendar
stage = "normalized_calendar"
print(f"\nStage: {stage}")
if checkpoint.stage_completed(stage):
    bars_1m = _load_or_none(checkpoint, stage)
else:
    bars_1m = None

if bars_1m is None:
    print("Normalizing calendar...")
    start = time.time()
    bars_1m = _load_or_none(checkpoint, "loaded_ohlc")
    if bars_1m is None:
        bars_1m = load_ohlc_frame(config)
        bars_1m = bars_1m[~bars_1m.index.duplicated(keep='first')]
    normalize_calendar(bars_1m, config)
    checkpoint.save_checkpoint(stage, bars_1m, timing=time.time() - start)

# 4. Validate bar sequence
stage = "validated_bars"
print(f"\nStage: {stage}")
if checkpoint.stage_completed(stage):
    bars_1m = _load_or_none(checkpoint, stage)
else:
    bars_1m = None

if bars_1m is None:
    print("Validating bar sequence...")
    start = time.time()
    bars_1m = _load_or_none(checkpoint, "normalized_calendar")
    if bars_1m is None:
        bars_1m = _load_or_none(checkpoint, "loaded_ohlc")
    if bars_1m is None:
        bars_1m = load_ohlc_frame(config)
        bars_1m = bars_1m[~bars_1m.index.duplicated(keep='first')]
        normalize_calendar(bars_1m, config)
    validate_bar_sequence(bars_1m, config)
    checkpoint.save_checkpoint(stage, bars_1m, timing=time.time() - start)

# 5. Resample to higher timeframes
stage = "resampled_timeframes"
print(f"\nStage: {stage}")
if checkpoint.stage_completed(stage):
    bars_by_tf = _load_or_none(checkpoint, stage)
else:
    bars_by_tf = None

if bars_by_tf is None:
    print("Resampling timeframes...")
    start = time.time()
    bars_1m = _load_or_none(checkpoint, "validated_bars")
    if bars_1m is None:
        raise ValueError("Cannot load validated_bars, previous stages failed")
    bars_by_tf = resample_timeframes(bars_1m, config)
    for tf, df in bars_by_tf.items():
        before = len(df)
        df = df[~df["end_ts"].duplicated(keep='first')]
        bars_by_tf[tf] = df
        if before != len(df):
            print(f"  {tf}: removed {before - len(df)} duplicates, {len(df)} bars")
        else:
            print(f"  {tf}: {len(df)} bars")
    checkpoint.save_checkpoint(stage, bars_by_tf, timing=time.time() - start)

# 6. Build features
stage = "built_features"
print(f"\nStage: {stage}")
if checkpoint.stage_completed(stage):
    features_by_tf = _load_or_none(checkpoint, stage)
else:
    features_by_tf = None

if features_by_tf is None:
    print("Building features...")
    start = time.time()
    bars_by_tf = _load_or_none(checkpoint, "resampled_timeframes")
    if bars_by_tf is None:
        raise ValueError("Cannot load resampled_timeframes, previous stages failed")
    features_by_tf = build_features(bars_by_tf, config)
    for tf, df in features_by_tf.items():
        print(f"  {tf}: {df.shape[1]} features")
    checkpoint.save_checkpoint(stage, features_by_tf, timing=time.time() - start)

# 7. Generate labels WITH MICRO-CHECKPOINTS
stage = "generated_labels"
print(f"\nStage: {stage}")
if checkpoint.stage_completed(stage):
    labels = _load_or_none(checkpoint, stage)
else:
    labels = None

if labels is None:
    start = time.time()
    bars_1m = _load_or_none(checkpoint, "validated_bars")
    if bars_1m is None:
        raise ValueError("Cannot load validated_bars, previous stages failed")
    labels = generate_labels_with_checkpoints(bars_1m, config, checkpoint_interval=1)
    for key, df in labels.items():
        horizon, threshold = key
        print(f"  Horizon {horizon}, threshold {threshold}: {len(df)} samples")
    checkpoint.save_checkpoint(stage, labels, timing=time.time() - start)

# 8. Create folds
stage = "created_folds"
print(f"\nStage: {stage}")
if checkpoint.stage_completed(stage):
    folds = _load_or_none(checkpoint, stage)
    label_df = None
else:
    folds = None

if folds is None:
    print("Creating walk-forward folds...")
    start = time.time()
    labels = _load_or_none(checkpoint, "generated_labels")
    if labels is None:
        raise ValueError("Cannot load generated_labels, previous stages failed")
    horizon_key = list(labels.keys())[0]
    label_df = labels[horizon_key]
    label_df = label_df[~label_df["ambiguous"]].copy()
    folds = build_walk_forward_folds(label_df, config)
    print(f"  Created {len(folds)} folds")
    checkpoint.save_checkpoint(stage, folds, timing=time.time() - start)
else:
    print("✓ Loaded folds from checkpoint")
    label_df = None

if label_df is None:
    labels = _load_or_none(checkpoint, "generated_labels")
    if labels is None:
        raise ValueError("Cannot load generated_labels")
    horizon_key = list(labels.keys())[0]
    label_df = labels[horizon_key]
    label_df = label_df[~label_df["ambiguous"]].copy()

# 9. Build windows WITH MICRO-CHECKPOINTS
stage = "built_windows"
print(f"\nStage: {stage}")
if checkpoint.stage_completed(stage):
    windows = _load_or_none(checkpoint, stage)
else:
    windows = None

if windows is None:
    start = time.time()
    lookbacks = {"1m": 90, "5m": 90, "15m": 90, "1h": 90, "4h": 90, "1d": 90}
    features_by_tf = _load_or_none(checkpoint, "built_features")
    if features_by_tf is None:
        raise ValueError("Cannot load built_features, previous stages failed")
    windows = build_windows_with_checkpoints(features_by_tf, lookbacks, checkpoint_interval=2)
    checkpoint.save_checkpoint(stage, windows, timing=time.time() - start)
else:
    print("✓ Loaded windows from checkpoint")

# 10. Align labels with window timestamps
stage = "aligned_labels"
print(f"\nStage: {stage}")
if checkpoint.stage_completed(stage):
    label_df_aligned = _load_or_none(checkpoint, stage)
else:
    label_df_aligned = None

if label_df_aligned is None:
    print("Aligning labels with windows...")
    start = time.time()
    window_ts_set = set(windows["1m"].reference_ts)
    label_df_aligned = label_df[label_df["reference_ts"].isin(window_ts_set)].reset_index(drop=True)
    print(f"  Aligned {len(label_df_aligned)} labels")
    checkpoint.save_checkpoint(stage, label_df_aligned, timing=time.time() - start)

# 11. Create dataset WITH MICRO-CHECKPOINTS
stage = "created_dataset"
print(f"\nStage: {stage}")
if checkpoint.stage_completed(stage):
    dataset = _load_or_none(checkpoint, stage)
    lookbacks = {"1m": 90, "5m": 90, "15m": 90, "1h": 90, "4h": 90, "1d": 90}
else:
    dataset = None

if dataset is None:
    start = time.time()
    lookbacks = {"1m": 90, "5m": 90, "15m": 90, "1h": 90, "4h": 90, "1d": 90}
    bars_1m = _load_or_none(checkpoint, "validated_bars")
    if bars_1m is None:
        raise ValueError("Cannot load validated_bars")
    dataset = create_dataset_with_checkpoints(windows, label_df_aligned, bars_1m, batch_size=100000)
    checkpoint.save_checkpoint(stage, dataset, timing=time.time() - start)
else:
    print("✓ Loaded dataset from checkpoint")

# 12. Create and save bundle
print(f"\nStage: saved_bundle")
print("Creating training bundle...")
folds = _load_or_none(checkpoint, "created_folds")
if folds is None:
    raise ValueError("Cannot load folds")

bundle = TrainingBundle(
    dataset=dataset,
    folds=tuple(folds),
    lookbacks_by_timeframe=lookbacks,
    retrieval_memory=tuple(),
)

stage = "saved_bundle"
print("Saving bundle...")
start = time.time()
bundle_path = Path("artifacts/training_bundle.pt")
bundle_path.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    "dataset": bundle.dataset,
    "folds": bundle.folds,
    "lookbacks_by_timeframe": bundle.lookbacks_by_timeframe,
    "retrieval_memory": bundle.retrieval_memory,
}, bundle_path)

checkpoint.save_checkpoint(stage, None, timing=time.time() - start)
print(f"\n✓ Training bundle saved to {bundle_path}")
print()
print(checkpoint.get_status())