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

# 8. Build windows
print("Building windows...")
lookbacks = {"1m": 90, "5m": 90, "15m": 90, "1h": 90, "4h": 90, "1d": 90}
windows = build_windows(features_by_tf, lookbacks_by_timeframe=lookbacks)

# 9. Create folds
print("Creating walk-forward folds...")
horizon_key = list(labels.keys())[0]
label_df = labels[horizon_key]
label_df = label_df[~label_df["ambiguous"]].copy()
folds = build_walk_forward_folds(label_df, config)
print(f"  Created {len(folds)} folds")

# 10. Align and create training dataset
print("Creating training dataset...")
# Align labels with windows
aligned_indices = label_df[label_df["reference_ts"].isin(windows["1m"].reference_ts)].index
label_df_aligned = label_df.loc[aligned_indices].reset_index(drop=True)

# Filter windows to match aligned labels
mask = [ts in label_df_aligned["reference_ts"].values for ts in windows["1m"].reference_ts]
windows_filtered = {
    tf: type(w)(windows[tf].windows[mask], windows[tf].reference_ts[mask], windows[tf].feature_names)
    for tf, w in windows.items()
}

dataset = TrainingDataset(
    reference_ts=tuple(windows_filtered["1m"].reference_ts),
    windows_by_timeframe={tf: w.windows for tf, w in windows_filtered.items()},
    reference_close=torch.tensor(
        bars_1m.loc[windows_filtered["1m"].reference_ts, "close"].values,
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

# 11. Create bundle
bundle = TrainingBundle(
    dataset=dataset,
    folds=tuple(folds),
    lookbacks_by_timeframe=lookbacks,
    retrieval_memory=tuple(),  # Empty initially
)

# 12. Save bundle
bundle_path = Path("artifacts/training_bundle.pt")
bundle_path.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    "dataset": bundle.dataset,
    "folds": bundle.folds,
    "lookbacks_by_timeframe": bundle.lookbacks_by_timeframe,
    "retrieval_memory": bundle.retrieval_memory,
}, bundle_path)

print(f"Training bundle saved to {bundle_path}")
