"""Feature registry for deterministic feature engineering (U3)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Iterable


class FeatureRegistryError(ValueError):
    """Raised when feature registry resolution fails."""


_ROLLING_PATTERN: Final[re.Pattern[str]] = re.compile(r"^roll_(mean|std|min|max)_(.+)_(\d+)$")

_GROUP_FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "returns": ("return", "log_return"),
    "ranges": ("hl_range", "oc_change", "body", "upper_wick", "lower_wick"),
    "wick_body_ratios": ("upper_wick_body_ratio", "lower_wick_body_ratio"),
    "calendar_time": ("dow", "hour", "minute", "dom", "month"),
}


@dataclass(frozen=True)
class ResolvedFeature:
    """Resolved feature definition."""

    name: str
    base_feature: str | None
    rolling_op: str | None
    window: int | None


def list_enabled_features(enabled_features: Iterable[str]) -> tuple[str, ...]:
    """Return enabled feature names in deterministic order.

    Args:
        enabled_features: Feature names from configuration.

    Returns:
        Expanded concrete feature names, preserving deterministic order.

    Raises:
        FeatureRegistryError: If the list is empty or contains duplicates.
    """
    enabled = tuple(enabled_features)
    if not enabled:
        raise FeatureRegistryError("enabled_features must not be empty.")
    if len(set(enabled)) != len(enabled):
        raise FeatureRegistryError("enabled_features must not contain duplicates.")

    expanded: list[str] = []
    for name in enabled:
        if name in _GROUP_FEATURES:
            expanded.extend(_GROUP_FEATURES[name])
        else:
            expanded.append(name)

    if len(set(expanded)) != len(expanded):
        raise FeatureRegistryError("Expanded enabled_features must not contain duplicates.")
    return tuple(expanded)


def resolve_feature(name: str) -> ResolvedFeature:
    """Resolve a feature name into a concrete feature definition.

    Supported rolling feature pattern:
    - roll_{op}_{base}_{window}
      where op in {mean,std,min,max} and window is a positive integer.

    Args:
        name: Feature name.

    Returns:
        A resolved feature.

    Raises:
        FeatureRegistryError: If the rolling feature is malformed.
    """
    match = _ROLLING_PATTERN.match(name)
    if match is None:
        return ResolvedFeature(name=name, base_feature=None, rolling_op=None, window=None)

    op, base, window_str = match.groups()
    window = int(window_str)
    if window <= 0:
        raise FeatureRegistryError(f"Rolling feature window must be positive: {name}")
    return ResolvedFeature(name=name, base_feature=base, rolling_op=op, window=window)
