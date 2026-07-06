"""Walk-forward fold generation utilities (U5)."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from training.config_schema import RuntimeConfig
from training.eval_contract import FoldSplit, WalkForwardFold

LOGGER = logging.getLogger(__name__)


class FoldError(ValueError):
    """Raised when fold generation fails."""


def build_walk_forward_folds(
    labeled_df: pd.DataFrame,
    config: RuntimeConfig,
) -> tuple[WalkForwardFold, ...]:
    """Build rolling walk-forward folds with strict temporal isolation.

    The fold builder uses the labeled dataset's `reference_ts` column as the
    canonical chronological timeline. It expects the dataset passed to U5 to be
    eligible for splitting (e.g., ambiguous tail rows must be explicitly handled
    by the caller).

    Args:
        labeled_df: Labeled dataset containing `reference_ts`.
        config: Validated runtime configuration (uses walk_forward settings).

    Returns:
        A tuple of walk-forward folds.

    Raises:
        FoldError: If the dataset is invalid, has ambiguous rows, or cannot be
            partitioned deterministically.
    """
    reference_ts = _extract_reference_ts(labeled_df)
    _fail_on_ambiguous(labeled_df)

    train_bars = int(config.walk_forward.train_bars)
    validation_bars = int(config.walk_forward.validation_bars)
    test_bars = int(config.walk_forward.test_bars)
    step_bars = int(config.walk_forward.step_bars)
    purge_gap_bars = int(max(config.labeling.horizon_bars))

    for name, value in (
        ("train_bars", train_bars),
        ("validation_bars", validation_bars),
        ("test_bars", test_bars),
        ("step_bars", step_bars),
        ("purge_gap_bars", purge_gap_bars),
    ):
        if value <= 0:
            raise FoldError(f"walk_forward.{name} must be positive.")

    n = len(reference_ts)
    fold_size = train_bars + purge_gap_bars + validation_bars + purge_gap_bars + test_bars
    if n < fold_size:
        LOGGER.info(
            "no_folds_generated",
            extra={
                "event": "no_folds_generated",
                "row_count": n,
                "required_rows": fold_size,
            },
        )
        return ()

    folds: list[WalkForwardFold] = []
    start = 0
    fold_id = 0
    while start + fold_size <= n:
        train_start = start
        train_end = train_start + train_bars
        val_start = train_end + purge_gap_bars
        val_end = val_start + validation_bars
        test_start = val_end + purge_gap_bars
        test_end = test_start + test_bars

        train_split = FoldSplit(
            start_index=train_start,
            end_index=train_end,
            start_ts=reference_ts[train_start],
            end_ts=reference_ts[train_end - 1],
        )
        validation_split = FoldSplit(
            start_index=val_start,
            end_index=val_end,
            start_ts=reference_ts[val_start],
            end_ts=reference_ts[val_end - 1],
        )
        test_split = FoldSplit(
            start_index=test_start,
            end_index=test_end,
            start_ts=reference_ts[test_start],
            end_ts=reference_ts[test_end - 1],
        )

        fold = WalkForwardFold(
            fold_id=fold_id,
            train=train_split,
            validation=validation_split,
            test=test_split,
            metadata={
                "train_bars": train_bars,
                "validation_bars": validation_bars,
                "test_bars": test_bars,
                "step_bars": step_bars,
                "purge_gap_bars": purge_gap_bars,
            },
        )
        folds.append(fold)
        fold_id += 1
        start += step_bars

    LOGGER.info(
        "built_walk_forward_folds",
        extra={
            "event": "built_walk_forward_folds",
            "fold_count": len(folds),
            "row_count": n,
            "train_bars": train_bars,
            "validation_bars": validation_bars,
            "test_bars": test_bars,
            "step_bars": step_bars,
            "purge_gap_bars": purge_gap_bars,
        },
    )
    return tuple(folds)


def validate_temporal_isolation(
    folds: Iterable[WalkForwardFold],
    reference_ts: pd.DatetimeIndex,
) -> None:
    """Validate strict temporal isolation and overlap prevention for folds.

    Args:
        folds: Fold definitions.
        reference_ts: Canonical sorted timestamp index used to build the folds.

    Raises:
        FoldError: If any fold violates chronology or overlap rules.
    """
    if reference_ts.tz is None:
        raise FoldError("reference_ts must be timezone-aware.")
    if reference_ts.has_duplicates:
        raise FoldError("reference_ts must not contain duplicates.")
    if not reference_ts.is_monotonic_increasing:
        raise FoldError("reference_ts must be monotonically increasing.")

    for fold in folds:
        train_idx = set(range(fold.train.start_index, fold.train.end_index))
        val_idx = set(range(fold.validation.start_index, fold.validation.end_index))
        test_idx = set(range(fold.test.start_index, fold.test.end_index))

        if train_idx & val_idx or train_idx & test_idx or val_idx & test_idx:
            raise FoldError(f"Fold {fold.fold_id} has overlapping indices across splits.")

        if not (fold.train.end_ts < fold.validation.start_ts < fold.test.start_ts):
            raise FoldError(f"Fold {fold.fold_id} violates strict chronological ordering.")
        purge_gap_bars = _require_int_metadata(fold.metadata, key="purge_gap_bars", default=0)
        if purge_gap_bars > 0:
            if fold.validation.start_index - fold.train.end_index < purge_gap_bars:
                raise FoldError(f"Fold {fold.fold_id} does not preserve the train/validation purge gap.")
            if fold.test.start_index - fold.validation.end_index < purge_gap_bars:
                raise FoldError(f"Fold {fold.fold_id} does not preserve the validation/test purge gap.")

        if fold.train.start_ts != reference_ts[fold.train.start_index]:
            raise FoldError(f"Fold {fold.fold_id} train start_ts mismatch.")
        if fold.test.end_ts != reference_ts[fold.test.end_index - 1]:
            raise FoldError(f"Fold {fold.fold_id} test end_ts mismatch.")


def _require_int_metadata(metadata: dict[str, object], *, key: str, default: int) -> int:
    if key not in metadata:
        return int(default)
    value = metadata[key]
    if value is None:
        return int(default)
    if isinstance(value, bool):
        raise FoldError(f"fold.metadata[{key}] must be an integer, not boolean.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise FoldError(f"fold.metadata[{key}] must be an integer.")


def _extract_reference_ts(labeled_df: pd.DataFrame) -> pd.DatetimeIndex:
    if "reference_ts" not in labeled_df.columns:
        raise FoldError("labeled_df must contain a reference_ts column.")
    reference_ts = pd.to_datetime(labeled_df["reference_ts"], errors="raise")
    index = pd.DatetimeIndex(reference_ts)
    if index.tz is None:
        raise FoldError("reference_ts must be timezone-aware.")
    if index.has_duplicates:
        raise FoldError("reference_ts must not contain duplicates.")
    if not index.is_monotonic_increasing:
        index = pd.DatetimeIndex(reference_ts.sort_values())
        if not index.is_monotonic_increasing:
            raise FoldError("reference_ts must be monotonically increasing.")
    return index


def _fail_on_ambiguous(labeled_df: pd.DataFrame) -> None:
    if "ambiguous" not in labeled_df.columns:
        return
    ambiguous = labeled_df["ambiguous"]
    if ambiguous.dtype != bool:
        ambiguous = ambiguous.astype(bool)
    if bool(ambiguous.any()):
        raise FoldError(
            "labeled_df contains ambiguous rows. Ambiguity handling must be explicit and resolved before fold splitting."
        )
