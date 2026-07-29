from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.sharded_store import ShardedDatasetStore


def test_sharded_store_slice_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "shards"
    (root / "windows").mkdir(parents=True)
    (root / "targets").mkdir(parents=True)
    (root / "reference").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
        (root / "windows" / tf).mkdir(parents=True)

    total = 3
    lookback = 2
    feature_dim = 4

    for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
        arr = np.arange(total * lookback * feature_dim, dtype=np.float32).reshape(total, lookback, feature_dim)
        np.save(root / "windows" / tf / "windows_shard_00000.npy", arr, allow_pickle=False)

    # Per-timeframe reference files
    for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
        np.save(root / "reference" / f"reference_close_{tf}_shard_00000.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32), allow_pickle=False)
    
    # Per-timeframe target files with multiple horizons
    horizons = [15, 60, 120]
    threshold = 10.0
    for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
        target_dict = {}
        for horizon in horizons:
            target_dict[f"event_flag_h{horizon}_t{threshold}"] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            target_dict[f"future_low_h{horizon}_t{threshold}"] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            target_dict[f"future_high_h{horizon}_t{threshold}"] = np.array([0.4, 0.5, 0.6], dtype=np.float32)
            target_dict[f"event_start_offset_h{horizon}_t{threshold}"] = np.array([1.0, 2.0, -1.0], dtype=np.float32)
            target_dict[f"maturity_offset_h{horizon}_t{threshold}"] = np.array([3.0, 4.0, -1.0], dtype=np.float32)
        np.savez_compressed(
            root / "targets" / f"targets_{tf}_shard_00000.npz",
            **target_dict
        )

    manifest = {
        "output_layout": {"root": str(root)},
        "lookbacks_by_timeframe": {tf: lookback for tf in ("1m", "5m", "15m", "1h", "4h", "1d")},
        "targets": {tf: {"total_samples": total, "shard_size": total, "num_shards": 1, "column_names": [f"event_flag_h{h}_t{threshold}" for h in horizons]} for tf in ("1m", "5m", "15m", "1h", "4h", "1d")},
        "windows": {tf: {"lookback": lookback, "num_features": feature_dim, "feature_names": ["a", "b", "c", "d"]} for tf in ("1m", "5m", "15m", "1h", "4h", "1d")},
        "label_selection": {"horizon": 15, "threshold": threshold},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Use default config (will use default horizons [15, 60, 120] and thresholds [10.0])
    store = ShardedDatasetStore(manifest_path)
    windows_by_tf, reference_close, targets = store.get_slice_unified(1, 3)

    assert reference_close.shape == (2,)
    assert float(reference_close[0].item()) == 2.0
    assert windows_by_tf["1m"].shape == (2, lookback, feature_dim)
    # Updated for horizon dimension: (batch, num_horizons)
    assert targets.event_flag["1m"].shape == (2, len(horizons))
    assert float(targets.event_flag["1m"][0, 0].item()) == 1.0  # First sample, first horizon

