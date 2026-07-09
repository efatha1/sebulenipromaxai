"""Minimal U1 entry points for config integration tests.

These wrappers exist to verify that the validated runtime configuration can be
consumed consistently by both training-oriented and inference-oriented entry
points without divergence.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from training.config_loader import load_config
from training.config_schema import RuntimeConfig


def load_training_config(config_path: str | Path, env: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Load validated config for the training entry point.

    Args:
        config_path: Path to the config file.
        env: Optional environment mapping.

    Returns:
        The validated runtime configuration.
    """
    return load_config(config_path=config_path, env=env)


def load_inference_config(config_path: str | Path, env: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Load validated config for the inference entry point.

    Args:
        config_path: Path to the config file.
        env: Optional environment mapping.

    Returns:
        The validated runtime configuration.
    """
    return load_config(config_path=config_path, env=env)

