"""Evaluation contract models for walk-forward folds (U5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


class EvalContractError(ValueError):
    """Raised when evaluation contract validation fails."""


@dataclass(frozen=True)
class FoldSplit:
    """A contiguous split inside a fold."""

    start_index: int
    end_index: int
    start_ts: datetime
    end_ts: datetime

    @property
    def size(self) -> int:
        return self.end_index - self.start_index


@dataclass(frozen=True)
class WalkForwardFold:
    """Definition of a single walk-forward fold."""

    fold_id: int
    train: FoldSplit
    validation: FoldSplit
    test: FoldSplit
    metadata: dict[str, object]


def fold_iterator(folds: Iterable[WalkForwardFold]) -> Iterable[WalkForwardFold]:
    """Yield folds in deterministic order.

    Args:
        folds: Fold definitions.

    Yields:
        Fold definitions in order.
    """
    yield from folds

