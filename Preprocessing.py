"""Kaggle-safe preprocessing pipeline (RAM-aware, deterministic).

This script builds the U2/U3/U4/U5 preprocessing artifacts from the configured
`1m` OHLC file and writes **sharded window tensors** to disk to avoid
materializing the full dataset in RAM.

Important notes:
- Paths are allowed to be relative to the project root (per operator preference).
- Kaggle-specific input/output paths can be configured either in YAML or by
  overriding `--output-root`.
- This script intentionally DOES NOT build an in-memory `TrainingDataset` or
  `training_bundle.pt` because that approach does not scale for multi-year 1m
  data under ~30GB RAM.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from training.config_loader import load_config
from training.data_loader import load_ohlc_frame
from training.data_quality import validate_bar_sequence
from training.features import build_features
from training.folds import build_walk_forward_folds
from training.labeling import generate_labels
from training.resample import resample_timeframes

LOGGER = logging.getLogger(__name__)

MODELED_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d")


@dataclass(frozen=True)
class OutputLayout:
    """Disk layout for preprocessing outputs."""

    root: Path

    @property
    def meta_dir(self) -> Path:
        return self.root / "meta"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def folds_path(self) -> Path:
        return self.root / "folds.json"

    @property
    def windows_dir(self) -> Path:
        return self.root / "windows"

    @property
    def targets_dir(self) -> Path:
        return self.root / "targets"

    @property
    def reference_dir(self) -> Path:
        return self.root / "reference"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.windows_dir.mkdir(parents=True, exist_ok=True)
        self.targets_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        for tf in MODELED_TIMEFRAMES:
            (self.windows_dir / tf).mkdir(parents=True, exist_ok=True)


def _configure_logging(*, verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _set_determinism(*, seed: int, allow_nondeterministic: bool) -> None:
    """Configure deterministic behavior (best-effort).

    This is primarily relevant once Torch enters the picture. Preprocessing is
    mostly pandas/numpy; we still seed numpy/torch for consistent sharding.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not allow_nondeterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _as_int64_ns(ts: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    idx = pd.DatetimeIndex(ts)
    # timezone-aware -> int64 ns since epoch
    return idx.view("int64")


def _build_window_view(values: np.ndarray, *, lookback: int) -> np.ndarray:
    """Return a strided view of trailing windows without copying the full tensor.

    Args:
        values: 2D array shaped (n_rows, n_features).
        lookback: Window length.

    Returns:
        A view shaped (n_windows, lookback, n_features).

    Raises:
        ValueError: If lookback is invalid.
    """
    if values.ndim != 2:
        raise ValueError("values must be 2D")
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    n_rows, n_features = values.shape
    if n_rows < lookback:
        raise ValueError(f"Not enough rows to build windows: n_rows={n_rows} lookback={lookback}")

    n_windows = n_rows - lookback + 1
    row_stride, col_stride = values.strides
    return np.lib.stride_tricks.as_strided(
        values,
        shape=(n_windows, lookback, n_features),
        strides=(row_stride, row_stride, col_stride),
        writeable=False,
    )


def _pick_label_frame(
    labels_by_key: dict[tuple[int, float], pd.DataFrame],
    *,
    preferred_horizon: int | None,
    preferred_threshold: float | None,
) -> tuple[tuple[int, float], pd.DataFrame]:
    keys = sorted(labels_by_key.keys(), key=lambda item: (int(item[0]), float(item[1])))
    if not keys:
        raise ValueError("No labels were generated (empty labels map).")

    if preferred_horizon is None and preferred_threshold is None:
        key = keys[0]
        return key, labels_by_key[key]

    for key in keys:
        horizon, threshold = key
        if preferred_horizon is not None and int(horizon) != int(preferred_horizon):
            continue
        if preferred_threshold is not None and float(threshold) != float(preferred_threshold):
            continue
        return key, labels_by_key[key]

    raise ValueError(
        "Requested label selection not found. "
        f"preferred_horizon={preferred_horizon} preferred_threshold={preferred_threshold} available={keys}"
    )


def _write_folds(layout: OutputLayout, folds: Any) -> None:
    payload = []
    for fold in folds:
        payload.append(
            {
                "fold_id": int(fold.fold_id),
                "train": {"start_ts": fold.train.start_ts.isoformat(), "end_ts": fold.train.end_ts.isoformat()},
                "validation": {
                    "start_ts": fold.validation.start_ts.isoformat(),
                    "end_ts": fold.validation.end_ts.isoformat(),
                },
                "test": {"start_ts": fold.test.start_ts.isoformat(), "end_ts": fold.test.end_ts.isoformat()},
            }
        )
    layout.folds_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_manifest(layout: OutputLayout, manifest: dict[str, Any]) -> None:
    layout.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_sharded_targets(
    layout: OutputLayout,
    *,
    label_df_aligned: pd.DataFrame,
    bars_1m: pd.DataFrame,
    shard_size: int,
) -> dict[str, Any]:
    """Write sharded targets + reference arrays.

    Returns:
        Manifest fragment with shard counts.
    """
    required_cols = ("reference_ts", "event_flag", "future_low", "future_high", "event_start_offset", "maturity_offset")
    missing = [col for col in required_cols if col not in label_df_aligned.columns]
    if missing:
        raise ValueError(f"Aligned label frame missing required columns: {missing}")

    total_samples = int(len(label_df_aligned))
    if total_samples == 0:
        raise ValueError("No aligned labels available (0 samples).")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")

    n_shards = int(math.ceil(total_samples / shard_size))
    ref_ts_ns = _as_int64_ns(label_df_aligned["reference_ts"])
    reference_close = bars_1m.loc[pd.DatetimeIndex(label_df_aligned["reference_ts"]), "close"].to_numpy(dtype=np.float32)

    for shard_id in range(n_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, total_samples)

        shard_ref_ts = ref_ts_ns[start:end]
        shard_close = reference_close[start:end]
        np.save(layout.reference_dir / f"reference_ts_ns_shard_{shard_id:05d}.npy", shard_ref_ts, allow_pickle=False)
        np.save(layout.reference_dir / f"reference_close_shard_{shard_id:05d}.npy", shard_close, allow_pickle=False)

        shard_targets = {
            "event_flag": label_df_aligned["event_flag"].iloc[start:end].to_numpy(dtype=np.float32),
            "future_low": label_df_aligned["future_low"].iloc[start:end].to_numpy(dtype=np.float32),
            "future_high": label_df_aligned["future_high"].iloc[start:end].to_numpy(dtype=np.float32),
            "event_start_offset": label_df_aligned["event_start_offset"].iloc[start:end].to_numpy(dtype=np.float32),
            "maturity_offset": label_df_aligned["maturity_offset"].iloc[start:end].to_numpy(dtype=np.float32),
        }
        np.savez_compressed(layout.targets_dir / f"targets_shard_{shard_id:05d}.npz", **shard_targets)

    return {"total_samples": total_samples, "shard_size": shard_size, "num_shards": n_shards}


def _write_sharded_windows(
    layout: OutputLayout,
    *,
    features_by_tf: dict[str, pd.DataFrame],
    label_df_aligned: pd.DataFrame,
    lookbacks_by_tf: dict[str, int],
    shard_size: int,
) -> dict[str, Any]:
    """Write sharded windows to disk timeframe-by-timeframe (RAM bounded)."""
    total_samples = int(len(label_df_aligned))
    n_shards = int(math.ceil(total_samples / shard_size))
    label_ts_ns = _as_int64_ns(label_df_aligned["reference_ts"])

    manifest_by_tf: dict[str, Any] = {}

    for tf in MODELED_TIMEFRAMES:
        tf_frame = features_by_tf[tf]
        lookback = int(lookbacks_by_tf[tf])

        tf_ref_ts = pd.DatetimeIndex(tf_frame.index[lookback - 1 :])
        tf_ref_ts_ns = _as_int64_ns(tf_ref_ts)
        if not np.all(tf_ref_ts_ns[1:] >= tf_ref_ts_ns[:-1]):
            raise ValueError(f"Feature index is not monotonic for timeframe={tf}.")

        # Map each label timestamp to the exact window reference index.
        # Because labels are computed on 1m and should align by end_ts, we require exact matches.
        positions = np.searchsorted(tf_ref_ts_ns, label_ts_ns, side="left")
        if positions.max(initial=0) >= len(tf_ref_ts_ns):
            raise ValueError(f"Label timestamps exceed available window references for timeframe={tf}.")
        if not np.array_equal(tf_ref_ts_ns[positions], label_ts_ns):
            # Fail early rather than silently aligning to the previous window.
            mismatches = np.flatnonzero(tf_ref_ts_ns[positions] != label_ts_ns)
            example = int(mismatches[0])
            raise ValueError(
                f"Timestamp alignment mismatch for timeframe={tf}. "
                f"example_label_ts_ns={int(label_ts_ns[example])} "
                f"matched_ts_ns={int(tf_ref_ts_ns[positions[example]])}"
            )

        # Build a strided view over all windows (small metadata object; no full tensor allocation).
        values = tf_frame.to_numpy(dtype=np.float32, copy=True)
        window_view = _build_window_view(values, lookback=lookback)

        for shard_id in range(n_shards):
            start = shard_id * shard_size
            end = min(start + shard_size, total_samples)
            idx = positions[start:end]

            shard_windows = window_view[idx].copy()  # materialize only the shard
            np.save(layout.windows_dir / tf / f"windows_shard_{shard_id:05d}.npy", shard_windows, allow_pickle=False)

            del shard_windows
            gc.collect()

        manifest_by_tf[tf] = {
            "lookback": lookback,
            "num_features": int(values.shape[1]),
            "feature_names": tuple(str(col) for col in tf_frame.columns),
            "num_window_references": int(len(tf_ref_ts)),
        }

        # Free timeframe memory before the next one.
        del window_view
        del values
        gc.collect()

    return manifest_by_tf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sebuleni preprocessing (RAM-aware, sharded).")
    parser.add_argument("--config-path", default="config/base.yaml", help="Path to runtime config YAML.")
    parser.add_argument(
        "--output-root",
        default=os.environ.get("SEBULENI_OUTPUT_ROOT", "artifacts/preprocessing_shards"),
        help="Root directory for preprocessing outputs.",
    )
    parser.add_argument("--lookback", type=int, default=90, help="Lookback for all timeframes (v1 default).")
    parser.add_argument("--shard-size", type=int, default=50_000, help="Samples per shard.")
    parser.add_argument("--label-horizon", type=int, default=None, help="Select a specific horizon for dataset shards.")
    parser.add_argument("--label-threshold", type=float, default=None, help="Select a specific threshold for dataset shards.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(verbose=bool(args.verbose))

    t0 = time.time()
    layout = OutputLayout(root=Path(args.output_root))
    layout.ensure()

    LOGGER.info("loading_config", extra={"event": "loading_config", "config_path": str(args.config_path)})
    config = load_config(config_path=args.config_path)
    _set_determinism(seed=int(config.training.random_seed), allow_nondeterministic=bool(config.training.allow_nondeterministic))

    lookback = int(args.lookback)
    lookbacks_by_tf = {tf: lookback for tf in MODELED_TIMEFRAMES}

    LOGGER.info("loading_ohlc")
    bars_1m = load_ohlc_frame(config)
    LOGGER.info("loaded_ohlc", extra={"event": "loaded_ohlc", "rows": int(len(bars_1m))})

    LOGGER.info("validating_bars")
    validate_bar_sequence(bars_1m, config)

    LOGGER.info("resampling_timeframes")
    bars_by_tf = resample_timeframes(bars_1m, config)

    LOGGER.info("building_features")
    features_by_tf = build_features(bars_by_tf, config)

    LOGGER.info("generating_labels_multi")
    labels_by_key = generate_labels(bars_1m, config, horizon_mode="multi")
    selected_key, label_df = _pick_label_frame(
        labels_by_key,
        preferred_horizon=args.label_horizon,
        preferred_threshold=args.label_threshold,
    )
    horizon, threshold = selected_key
    LOGGER.info(
        "selected_label_frame",
        extra={"event": "selected_label_frame", "horizon": int(horizon), "threshold": float(threshold), "rows": int(len(label_df))},
    )

    label_df = label_df[~label_df["ambiguous"]].copy()
    if len(label_df) == 0:
        raise ValueError("All labels are ambiguous after filtering; cannot proceed.")

    # Align labels to 1m window reference timestamps deterministically.
    ref_ts_1m = pd.DatetimeIndex(features_by_tf["1m"].index[lookback - 1 :])
    mask = pd.DatetimeIndex(label_df["reference_ts"]).isin(ref_ts_1m)
    label_df_aligned = label_df.loc[mask].reset_index(drop=True)
    if len(label_df_aligned) == 0:
        raise ValueError("No labels align to available 1m window reference timestamps.")

    # Persist aligned labels for traceability/debugging.
    label_out = layout.labels_dir / f"labels_h{int(horizon)}_t{float(threshold)}.parquet"
    label_df_aligned.to_parquet(label_out, index=False)

    LOGGER.info("building_folds")
    folds = build_walk_forward_folds(label_df_aligned, config)
    _write_folds(layout, folds)

    LOGGER.info("writing_sharded_targets")
    target_manifest = _write_sharded_targets(
        layout,
        label_df_aligned=label_df_aligned,
        bars_1m=bars_1m,
        shard_size=int(args.shard_size),
    )

    LOGGER.info("writing_sharded_windows")
    windows_manifest = _write_sharded_windows(
        layout,
        features_by_tf=features_by_tf,
        label_df_aligned=label_df_aligned,
        lookbacks_by_tf=lookbacks_by_tf,
        shard_size=int(args.shard_size),
    )

    manifest = {
        "created_at_unix": int(time.time()),
        "config": {
            "instrument_id": config.instrument.instrument_id,
            "ohlc_path": str(config.data_source.ohlc_path),
            "runtime_timezone": config.time.runtime_timezone,
        },
        "label_selection": {"horizon": int(horizon), "threshold": float(threshold)},
        "lookbacks_by_timeframe": lookbacks_by_tf,
        "targets": target_manifest,
        "windows": windows_manifest,
        "output_layout": {"root": str(layout.root)},
        "elapsed_seconds": float(time.time() - t0),
    }
    _write_manifest(layout, manifest)

    LOGGER.warning("preprocessing_complete output_root=%s elapsed=%.1fs", str(layout.root), time.time() - t0)


if __name__ == "__main__":
    main()
