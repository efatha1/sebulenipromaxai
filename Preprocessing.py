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
from training.labeling import generate_labels, generate_labels_multi_timeframe
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


def _compute_common_history_start(
    *,
    features_by_tf: dict[str, pd.DataFrame],
    lookbacks_by_tf: dict[str, int],
) -> pd.Timestamp:
    """Compute the first reference timestamp where all timeframes have full history.

    Each timeframe requires `lookback` bars to construct a trailing window ending
    at a reference timestamp. Therefore, the first eligible reference timestamp
    for a timeframe is `feature_index[lookback - 1]`.

    For multi-timeframe modeling, we must restrict samples to timestamps that
    have sufficient history across *all* modeled timeframes.

    Args:
        features_by_tf: Mapping timeframe -> feature frame indexed by `end_ts`.
        lookbacks_by_tf: Mapping timeframe -> lookback.

    Returns:
        The common start timestamp (inclusive) where all timeframes have
        sufficient history.

    Raises:
        ValueError: If any timeframe is missing, lookback is invalid, index is not
            timezone-aware, or no common start exists.
    """
    starts: list[pd.Timestamp] = []
    for tf in MODELED_TIMEFRAMES:
        if tf not in features_by_tf:
            raise ValueError(f"features_by_tf missing timeframe={tf}.")
        if tf not in lookbacks_by_tf:
            raise ValueError(f"lookbacks_by_tf missing timeframe={tf}.")
        lookback = int(lookbacks_by_tf[tf])
        if lookback <= 0:
            raise ValueError(f"lookback must be positive for timeframe={tf}.")
        frame = features_by_tf[tf]
        if len(frame) < lookback:
            raise ValueError(
                f"Not enough rows to build windows for timeframe={tf}: rows={len(frame)} lookback={lookback}"
            )
        idx = pd.DatetimeIndex(frame.index)
        if idx.tz is None:
            raise ValueError(f"Feature index must be timezone-aware for timeframe={tf}.")
        starts.append(pd.Timestamp(idx[lookback - 1]))
    return max(starts)


def _resolve_lookbacks_by_timeframe(*, config: Any, fallback_lookback: int) -> dict[str, int]:
    """Resolve per-timeframe lookbacks from config or global fallback.

    Precedence:
    1. `config.preprocessing.lookbacks_by_timeframe` when configured.
    2. Global fallback lookback applied to all modeled timeframes.

    Args:
        config: Validated runtime config-like object.
        fallback_lookback: Fallback lookback to apply to all timeframes.

    Returns:
        Mapping timeframe -> positive lookback.

    Raises:
        ValueError: If the fallback is invalid or the resolved mapping is incomplete.
    """
    if fallback_lookback <= 0:
        raise ValueError("fallback_lookback must be positive.")
    preprocessing = getattr(config, "preprocessing", None)
    configured = None if preprocessing is None else getattr(preprocessing, "lookbacks_by_timeframe", None)
    if configured is not None:
        resolved = {str(tf): int(lookback) for tf, lookback in configured.items()}
    else:
        resolved = {tf: int(fallback_lookback) for tf in MODELED_TIMEFRAMES}

    if set(resolved.keys()) != set(MODELED_TIMEFRAMES):
        raise ValueError(f"Resolved lookbacks must contain exactly {MODELED_TIMEFRAMES}.")
    for tf, lookback in resolved.items():
        if int(lookback) <= 0:
            raise ValueError(f"lookback must be positive for timeframe={tf}.")
    return resolved


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


def _write_sharded_targets_unified(
    layout: OutputLayout,
    *,
    labels_by_key: dict[tuple[str, int, float], pd.DataFrame],
    bars_by_timeframe: dict[str, pd.DataFrame],
    shard_size: int,
) -> dict[str, Any]:
    """Write sharded targets for all 18 (timeframe, horizon, threshold) combinations.

    Uses mixed precision:
    - event_flag: float32 (critical for binary classification)
    - future_low, future_high: float16 (price data, acceptable precision loss)
    - event_start_offset, maturity_offset: float16 (offsets, acceptable precision loss)

    Storage structure (Option A - per-timeframe storage):
    - Separate targets_{timeframe}_shard_{shard_id:05d}.npz file per timeframe per shard
    - Column naming convention: {field}_h{horizon}_t{threshold} (timeframe in filename)
      Example: targets_1m_shard_00000.npz contains event_flag_h15_t10.0, future_low_h15_t10.0, etc.
    - Per-file columns: 5 fields × 3 horizons × 1 threshold = 15 columns per timeframe
    - Total files: 6 timeframes × n_shards

    Args:
        layout: Output layout for disk paths.
        labels_by_key: Mapping (timeframe, horizon, threshold) -> label DataFrame.
        bars_by_timeframe: Mapping timeframe -> OHLC bars for reference close prices.
        shard_size: Samples per shard.

    Returns:
        Manifest fragment with shard counts and column metadata per timeframe.
    """
    # Group labels by timeframe
    labels_by_timeframe: dict[str, dict[tuple[int, float], pd.DataFrame]] = {}
    for key, labels_df in labels_by_key.items():
        timeframe, horizon, threshold = key
        if timeframe not in labels_by_timeframe:
            labels_by_timeframe[timeframe] = {}
        labels_by_timeframe[timeframe][(horizon, threshold)] = labels_df

    # Process each timeframe separately
    timeframe_manifests = {}
    for timeframe in sorted(labels_by_timeframe.keys()):
        timeframe_labels = labels_by_timeframe[timeframe]
        
        # Filter ambiguous labels for this timeframe
        first_key = next(iter(timeframe_labels.keys()))
        first_df = timeframe_labels[first_key]
        first_df_filtered = first_df[~first_df["ambiguous"]].copy()
        
        if len(first_df_filtered) == 0:
            raise ValueError(f"All labels are ambiguous for timeframe {timeframe} after filtering; cannot proceed.")
        
        # Use this timeframe's native timestamps
        ref_ts = pd.DatetimeIndex(first_df_filtered["reference_ts"])
        total_samples = int(len(ref_ts))
        
        if total_samples == 0:
            raise ValueError(f"No labels available for timeframe {timeframe} (0 samples).")
        
        n_shards = int(math.ceil(total_samples / shard_size))
        ref_ts_ns = _as_int64_ns(ref_ts)
        
        # Get reference close from this timeframe's bars
        bars_tf = bars_by_timeframe[timeframe]
        reference_close = bars_tf.loc[ref_ts, "close"].to_numpy(dtype=np.float32)
        
        # Build column name mapping for this timeframe
        column_names = []
        for (horizon, threshold) in sorted(timeframe_labels.keys()):
            for field in ("event_flag", "future_low", "future_high", "event_start_offset", "maturity_offset"):
                col_name = f"{field}_h{horizon}_t{threshold}"
                column_names.append((col_name, (horizon, threshold), field))
        
        # Write shards for this timeframe
        for shard_id in range(n_shards):
            start = shard_id * shard_size
            end = min(start + shard_size, total_samples)
            
            # Write reference arrays for this timeframe
            shard_ref_ts = ref_ts_ns[start:end]
            shard_close = reference_close[start:end]
            np.save(layout.reference_dir / f"reference_ts_ns_{timeframe}_shard_{shard_id:05d}.npy", shard_ref_ts, allow_pickle=False)
            np.save(layout.reference_dir / f"reference_close_{timeframe}_shard_{shard_id:05d}.npy", shard_close, allow_pickle=False)
            
            # Build targets dictionary for this timeframe
            shard_targets = {}
            for col_name, (horizon, threshold), field in column_names:
                labels_df = timeframe_labels[(horizon, threshold)]
                # Filter ambiguous labels
                labels_df_filtered = labels_df[~labels_df["ambiguous"]].copy()
                # Get values for this shard
                values = labels_df_filtered[field].iloc[start:end].to_numpy()
                
                # Apply mixed precision
                if field == "event_flag":
                    shard_targets[col_name] = values.astype(np.float32)
                else:
                    shard_targets[col_name] = values.astype(np.float16)
            
            np.savez_compressed(layout.targets_dir / f"targets_{timeframe}_shard_{shard_id:05d}.npz", **shard_targets)
        
        timeframe_manifests[timeframe] = {
            "total_samples": total_samples,
            "shard_size": shard_size,
            "num_shards": n_shards,
            "column_names": [col_name for col_name, _, _ in column_names],
            "num_columns": len(column_names),
        }
    
    return timeframe_manifests


def _write_sharded_windows(
    layout: OutputLayout,
    *,
    features_by_tf: dict[str, pd.DataFrame],
    labels_by_key: dict[tuple[str, int, float], pd.DataFrame],
    lookbacks_by_tf: dict[str, int],
    shard_size: int,
) -> dict[str, Any]:
    """Write sharded windows to disk timeframe-by-timeframe (RAM bounded).

    With Option A (per-timeframe storage), each timeframe uses its own native timestamps
    for window construction, avoiding cross-timeframe alignment issues.
    """
    # Group labels by timeframe to get native timestamps
    labels_by_timeframe: dict[str, dict[tuple[int, float], pd.DataFrame]] = {}
    for key, labels_df in labels_by_key.items():
        timeframe, horizon, threshold = key
        if timeframe not in labels_by_timeframe:
            labels_by_timeframe[timeframe] = {}
        labels_by_timeframe[timeframe][(horizon, threshold)] = labels_df

    manifest_by_tf: dict[str, Any] = {}

    for tf in MODELED_TIMEFRAMES:
        tf_frame = features_by_tf[tf]
        lookback = int(lookbacks_by_tf[tf])

        tf_ref_ts = pd.DatetimeIndex(tf_frame.index[lookback - 1 :])
        tf_ref_ts_ns = _as_int64_ns(tf_ref_ts)
        if not np.all(tf_ref_ts_ns[1:] >= tf_ref_ts_ns[:-1]):
            raise ValueError(f"Feature index is not monotonic for timeframe={tf}.")

        # Get this timeframe's native label timestamps
        if tf not in labels_by_timeframe:
            raise ValueError(f"No labels available for timeframe={tf}")
        
        first_key = next(iter(labels_by_timeframe[tf].keys()))
        tf_labels = labels_by_timeframe[tf][first_key]
        tf_labels_filtered = tf_labels[~tf_labels["ambiguous"]].copy()
        
        # Align labels to this timeframe's feature timestamps
        # Use the same alignment logic as the main flow
        tf_label_ts = pd.DatetimeIndex(tf_labels_filtered["reference_ts"])
        mask = tf_label_ts.isin(tf_ref_ts)
        tf_labels_aligned = tf_labels_filtered.loc[mask].reset_index(drop=True)
        
        if len(tf_labels_aligned) == 0:
            raise ValueError(f"No labels align to available {tf} window reference timestamps.")
        
        tf_label_ts_ns = _as_int64_ns(tf_labels_aligned["reference_ts"])
        
        total_samples = int(len(tf_label_ts_ns))
        n_shards = int(math.ceil(total_samples / shard_size))

        # Verify exact timestamp matches after alignment
        positions = np.searchsorted(tf_ref_ts_ns, tf_label_ts_ns, side="left")
        if positions.max(initial=0) >= len(tf_ref_ts_ns):
            raise ValueError(f"Label timestamps exceed available window references for timeframe={tf}.")
        if not np.array_equal(tf_ref_ts_ns[positions], tf_label_ts_ns):
            mismatches = np.flatnonzero(tf_ref_ts_ns[positions] != tf_label_ts_ns)
            example = int(mismatches[0])
            raise ValueError(
                f"Timestamp alignment mismatch for timeframe={tf}. "
                f"example_label_ts_ns={int(tf_label_ts_ns[example])} "
                f"matched_ts_ns={int(tf_ref_ts_ns[positions[example]])}"
            )

        # Build a strided view over all windows (small metadata object; no full tensor allocation).
        values = tf_frame.to_numpy(dtype=np.float32, copy=False)
        if not values.flags["C_CONTIGUOUS"]:
            # Strided windows require predictable memory layout; enforce contiguous storage.
            values = np.ascontiguousarray(values)
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
    parser = argparse.ArgumentParser(description="Sebuleni preprocessing (RAM-aware, sharded, unified multi-timeframe).")
    parser.add_argument("--config-path", default="config/base.yaml", help="Path to runtime config YAML.")
    parser.add_argument(
        "--preprocessing-output-root",
        default=os.environ.get("SEBULENI__PREPROCESSING__OUTPUT_ROOT"),
        help="Root directory for preprocessing outputs (overrides config).",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("SEBULENI_OUTPUT_ROOT"),
        help="Root directory for preprocessing outputs (deprecated, use --preprocessing-output-root).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=90,
        help="Fallback lookback for all timeframes when preprocessing.lookbacks_by_timeframe is not configured.",
    )
    parser.add_argument("--shard-size", type=int, default=50_000, help="Samples per shard.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(verbose=bool(args.verbose))

    LOGGER.info("loading_config", extra={"event": "loading_config", "config_path": str(args.config_path)})
    config = load_config(config_path=args.config_path)

    # Resolve output root with precedence: CLI flag > config > env var > default
    output_root = None
    if args.preprocessing_output_root is not None:
        output_root = Path(args.preprocessing_output_root)
    elif args.output_root is not None:
        import warnings
        warnings.warn(
            "--output-root is deprecated. Use --preprocessing-output-root instead.",
            DeprecationWarning,
            stacklevel=2
        )
        output_root = Path(args.output_root)
    elif config.preprocessing and config.preprocessing.output_root is not None:
        output_root = config.preprocessing.output_root
    else:
        output_root = Path("artifacts/preprocessing_shards")

    t0 = time.time()
    layout = OutputLayout(root=output_root)
    layout.ensure()

    _set_determinism(seed=int(config.training.random_seed), allow_nondeterministic=bool(config.training.allow_nondeterministic))

    lookbacks_by_tf = _resolve_lookbacks_by_timeframe(config=config, fallback_lookback=int(args.lookback))
    LOGGER.info(
        "resolved_lookbacks_by_timeframe",
        extra={"event": "resolved_lookbacks_by_timeframe", "lookbacks_by_timeframe": dict(lookbacks_by_tf)},
    )

    LOGGER.info("loading_ohlc")
    bars_1m = load_ohlc_frame(config)
    LOGGER.info("loaded_ohlc", extra={"event": "loaded_ohlc", "rows": int(len(bars_1m))})

    LOGGER.info("validating_bars")
    validate_bar_sequence(bars_1m, config)

    LOGGER.info("resampling_timeframes")
    bars_by_tf = resample_timeframes(bars_1m, config)

    LOGGER.info("building_features")
    features_by_tf = build_features(bars_by_tf, config)

    LOGGER.info("generating_labels_multi_timeframe")
    labels_by_key = generate_labels_multi_timeframe(bars_by_tf, config, horizon_mode="multi")
    
    # Use the first label frame for alignment (all have same reference timestamps)
    first_key = next(iter(labels_by_key.keys()))
    label_df = labels_by_key[first_key]
    
    label_df = label_df[~label_df["ambiguous"]].copy()
    if len(label_df) == 0:
        raise ValueError("All labels are ambiguous after filtering; cannot proceed.")

    # Align labels to 1m window reference timestamps deterministically.
    ref_ts_1m = pd.DatetimeIndex(features_by_tf["1m"].index[lookbacks_by_tf["1m"] - 1 :])
    mask = pd.DatetimeIndex(label_df["reference_ts"]).isin(ref_ts_1m)
    label_df_aligned = label_df.loc[mask].reset_index(drop=True)
    if len(label_df_aligned) == 0:
        raise ValueError("No labels align to available 1m window reference timestamps.")

    # Ensure each label timestamp has sufficient lookback history across all modeled timeframes.
    # Note: Without this, higher-timeframe windows (e.g. 5m lookback=90) cannot be constructed
    # for early 1m timestamps even though 1m windows exist.
    common_start = _compute_common_history_start(features_by_tf=features_by_tf, lookbacks_by_tf=lookbacks_by_tf)
    before = int(len(label_df_aligned))
    label_df_aligned = label_df_aligned.loc[pd.DatetimeIndex(label_df_aligned["reference_ts"]) >= common_start].reset_index(
        drop=True
    )
    dropped = before - int(len(label_df_aligned))
    LOGGER.info(
        "filtered_labels_for_common_history",
        extra={
            "event": "filtered_labels_for_common_history",
            "common_start": common_start.isoformat(),
            "dropped": int(dropped),
            "remaining": int(len(label_df_aligned)),
        },
    )
    if len(label_df_aligned) == 0:
        raise ValueError(
            "All labels were filtered out due to insufficient multi-timeframe history. "
            f"common_start={common_start.isoformat()} lookbacks_by_tf={lookbacks_by_tf}"
        )

    LOGGER.info("building_folds")
    folds = build_walk_forward_folds(label_df_aligned, config)
    _write_folds(layout, folds)

    LOGGER.info("writing_sharded_targets_unified")
    target_manifest = _write_sharded_targets_unified(
        layout,
        labels_by_key=labels_by_key,
        bars_by_timeframe=bars_by_tf,
        shard_size=int(args.shard_size),
    )

    LOGGER.info("writing_sharded_windows")
    windows_manifest = _write_sharded_windows(
        layout,
        features_by_tf=features_by_tf,
        labels_by_key=labels_by_key,
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
        "lookbacks_by_timeframe": lookbacks_by_tf,
        "targets": target_manifest,  # Now a dict of manifests per timeframe
        "windows": windows_manifest,
        "output_layout": {"root": str(layout.root)},
        "elapsed_seconds": float(time.time() - t0),
    }
    _write_manifest(layout, manifest)

    LOGGER.warning("preprocessing_complete output_root=%s elapsed=%.1fs", str(layout.root), time.time() - t0)


if __name__ == "__main__":
    main()
