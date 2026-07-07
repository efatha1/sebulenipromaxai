"""Preprocessing pipeline from raw OHLC to training bundle."""
import torch
from pathlib import Path
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

# 1. Load config
config = load_config("config/base.yaml")

# 2. Load OHLC data
print("Loading OHLC data...")
bars_1m = load_ohlc_frame(config)
print(f"Loaded {len(bars_1m)} 1-minute bars")

# Remove duplicates from 1m data
before = len(bars_1m)
bars_1m = bars_1m[~bars_1m.index.duplicated(keep='first')]
print(f"Removed {before - len(bars_1m)} duplicate 1m bars, {len(bars_1m)} remaining")

# 3. Normalize calendar
print("Normalizing calendar...")
normalize_calendar(bars_1m, config)

# 4. Validate bar sequence
print("Validating bar sequence...")
validate_bar_sequence(bars_1m, config)

# 5. Resample to higher timeframes
print("Resampling timeframes...")
bars_by_tf = resample_timeframes(bars_1m, config)
# Remove duplicates from resampled data
for tf, df in bars_by_tf.items():
    before = len(df)
    df = df[~df["end_ts"].duplicated(keep='first')]
    bars_by_tf[tf] = df
    if before != len(df):
        print(f"  {tf}: removed {before - len(df)} duplicates, {len(df)} bars remaining")
    else:
        print(f"  {tf}: {len(df)} bars")

# 6. Build features
print("Building features...")
features_by_tf = build_features(bars_by_tf, config)
for tf, df in features_by_tf.items():
    print(f"  {tf}: {df.shape[1]} features")

# 7. Generate labels
print("Generating labels...")
labels = generate_labels(bars_1m, config, horizon_mode="multi")
for key, df in labels.items():
    horizon, threshold = key
    print(f"  Horizon {horizon}, threshold {threshold}: {len(df)} samples")

# 8. Create folds
print("Creating walk-forward folds...")
horizon_key = list(labels.keys())[0]
label_df = labels[horizon_key]
label_df = label_df[~label_df["ambiguous"]].copy()
folds = build_walk_forward_folds(label_df, config)
print(f"  Created {len(folds)} folds")

# 9. Build windows FIRST (before alignment)
print("Building windows...")
lookbacks = {"1m": 90, "5m": 90, "15m": 90, "1h": 90, "4h": 90, "1d": 90}
windows = build_windows(features_by_tf, lookbacks_by_timeframe=lookbacks)

# 10. Align labels with window timestamps
print("Aligning labels with windows...")
window_ts_set = set(windows["1m"].reference_ts)
label_df_aligned = label_df[label_df["reference_ts"].isin(window_ts_set)].reset_index(drop=True)
print(f"  Aligned {len(label_df_aligned)} labels to {len(window_ts_set)} window timestamps")

# 11. Create index mapping for windows that match aligned labels
print("Creating training dataset...")
# Build indices: find which window indices correspond to our aligned labels
window_ts_list = list(windows["1m"].reference_ts)
aligned_indices = [window_ts_list.index(ts) for ts in label_df_aligned["reference_ts"]]

# Slice all windows and targets using these indices
windows_filtered = {
    tf: windows[tf].windows[aligned_indices]
    for tf in windows.keys()
}

dataset = TrainingDataset(
    reference_ts=tuple(label_df_aligned["reference_ts"]),
    windows_by_timeframe=windows_filtered,
    reference_close=torch.tensor(
        [bars_1m.loc[ts, "close"] for ts in label_df_aligned["reference_ts"]],
        dtype=torch.float32
    ),
    targets=MultiTaskTargets(
        event_flag=torch.tensor(label_df_aligned["event_flag"].values, dtype=torch.float32),
        future_low=torch.tensor(label_df_aligned["future_low"].values, dtype=torch.float32),
        future_high=torch.tensor(label_df_aligned["future_high"].values, dtype=torch.float32),
        event_start_offset=torch.tensor(label_df_aligned["event_start_offset"].values, dtype=torch.float32),
        maturity_offset=torch.tensor(label_df_aligned["maturity_offset"].values, dtype=torch.float32),
        confidence_target=torch.tensor([0.5] * len(label_df_aligned), dtype=torch.float32),
        regime_target=None,
    ),
)

# 12. Create bundle
bundle = TrainingBundle(
    dataset=dataset,
    folds=tuple(folds),
    lookbacks_by_timeframe=lookbacks,
    retrieval_memory=tuple(),
)

# 13. Save bundle
bundle_path = Path("artifacts/training_bundle.pt")
bundle_path.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    "dataset": bundle.dataset,
    "folds": bundle.folds,
    "lookbacks_by_timeframe": bundle.lookbacks_by_timeframe,
    "retrieval_memory": bundle.retrieval_memory,
}, bundle_path)

print(f"Training bundle saved to {bundle_path}")

print(f"Training bundle saved to {bundle_path}")
