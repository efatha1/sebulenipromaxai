"""Configuration loading and validation for Sebuleni Pro Max AI."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from training.config_schema import RuntimeConfig

LOGGER = logging.getLogger(__name__)
ENV_PREFIX = "SEBULENI__"


class ConfigError(ValueError):
    """Base configuration exception."""


class ConfigFileNotFoundError(ConfigError):
    """Raised when the requested config file does not exist."""


class ConfigValidationError(ConfigError):
    """Raised when config validation fails."""


class ConfigOverrideError(ConfigError):
    """Raised when an environment override is malformed or unknown."""


def validate_config(raw_config: Mapping[str, Any]) -> RuntimeConfig:
    """Validate raw config data and return an immutable runtime config.

    Args:
        raw_config: Raw config mapping to validate.

    Returns:
        The validated immutable runtime configuration.

    Raises:
        ConfigValidationError: If the input is not a mapping or schema
            validation fails.
    """
    if not isinstance(raw_config, Mapping):
        raise ConfigValidationError("Configuration payload must be a mapping.")

    try:
        config = RuntimeConfig.model_validate(dict(raw_config))
    except ValidationError as exc:
        formatted_errors = "; ".join(_format_validation_error(error) for error in exc.errors())
        raise ConfigValidationError(f"Configuration validation failed: {formatted_errors}") from exc

    LOGGER.info(
        "validated_runtime_config",
        extra={
            "event": "validated_runtime_config",
            "instrument_id": config.instrument.instrument_id,
            "target_timeframes": list(config.resampling.target_timeframes),
            "device_preference": config.training.device_preference,
        },
    )
    return config


def load_config(config_path: str | Path, env: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Load configuration from YAML and deterministic environment overrides.

    Args:
        config_path: Path to the YAML configuration file.
        env: Optional environment mapping. If omitted, `os.environ` is used.

    Returns:
        The validated immutable runtime configuration.

    Raises:
        ConfigFileNotFoundError: If the config file does not exist.
        ConfigValidationError: If the YAML is malformed or fails validation.
        ConfigOverrideError: If an environment override is malformed.
    """
    path = Path(config_path)
    if not str(path).strip():
        raise ConfigValidationError("Configuration path must not be empty.")
    if not path.exists():
        raise ConfigFileNotFoundError(f"Configuration file does not exist: {path}")
    if not path.is_file():
        raise ConfigValidationError(f"Configuration path must point to a file: {path}")

    data = _read_yaml_file(path)
    overridden_data = _apply_environment_overrides(data, env)

    LOGGER.info(
        "loaded_config_payload",
        extra={"event": "loaded_config_payload", "config_path": str(path)},
    )
    return validate_config(overridden_data)


def _read_yaml_file(config_path: Path) -> dict[str, Any]:
    """Read and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed YAML mapping.

    Raises:
        ConfigValidationError: If the file is unreadable, malformed, or not a
            mapping.
    """
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigValidationError(f"Failed to read configuration file: {config_path}") from exc

    if not raw_text.strip():
        raise ConfigValidationError(f"Configuration file is empty: {config_path}")

    try:
        loaded = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Configuration YAML is malformed: {config_path}") from exc

    if not isinstance(loaded, dict):
        raise ConfigValidationError("Top-level configuration must be a mapping.")
    return loaded


def _apply_environment_overrides(
    config_data: Mapping[str, Any],
    env: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Apply deterministic environment overrides.

    Overrides use `SEBULENI__SECTION__FIELD=value` style keys. Nested fields are
    resolved case-insensitively against the existing config shape. Unknown keys
    fail fast instead of being silently added.

    Args:
        config_data: Parsed config mapping.
        env: Environment mapping or `None`.

    Returns:
        A deep-copied config mapping with overrides applied.

    Raises:
        ConfigOverrideError: If an override path is malformed or unknown.
    """
    runtime_env = dict(env or os.environ)
    overridden = deepcopy(dict(config_data))

    for key in sorted(runtime_env):
        if not key.startswith(ENV_PREFIX):
            continue
        override_path = key[len(ENV_PREFIX) :].split("__")
        if not override_path or any(not part for part in override_path):
            raise ConfigOverrideError(f"Invalid environment override key: {key}")

        parsed_value = _parse_override_value(runtime_env[key])
        _set_nested_value(overridden, override_path, parsed_value, env_key=key)

    return overridden


def _parse_override_value(raw_value: str) -> Any:
    """Parse an override value deterministically.

    Args:
        raw_value: Raw environment value.

    Returns:
        A parsed scalar, list, or mapping.
    """
    lowered = raw_value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _set_nested_value(
    config_data: dict[str, Any],
    path_parts: list[str],
    value: Any,
    *,
    env_key: str,
) -> None:
    """Set a nested config value while forbidding unknown paths.

    Args:
        config_data: Config mapping to update.
        path_parts: Normalized override path parts.
        value: Parsed override value.
        env_key: Source environment key for diagnostics.

    Raises:
        ConfigOverrideError: If the path is unknown or does not resolve to a
            mapping structure.
    """
    current: dict[str, Any] = config_data
    for part in path_parts[:-1]:
        actual_key = _resolve_key(current, part, env_key)
        next_value = current[actual_key]
        if not isinstance(next_value, dict):
            joined = "__".join(path_parts)
            raise ConfigOverrideError(
                f"Environment override path does not resolve to a mapping: {env_key} ({joined})"
            )
        current = next_value

    final_key = _resolve_key(current, path_parts[-1], env_key)
    current[final_key] = value


def _resolve_key(current: Mapping[str, Any], lookup_key: str, env_key: str) -> str:
    """Resolve a config key case-insensitively against an existing mapping.

    Args:
        current: Current mapping level.
        lookup_key: Requested key part.
        env_key: Source environment key for diagnostics.

    Returns:
        The actual key present in the mapping.

    Raises:
        ConfigOverrideError: If the key does not exist.
    """
    lowered_lookup = lookup_key.lower()
    for actual_key in current:
        if actual_key.lower() == lowered_lookup:
            return actual_key

    allowed = ", ".join(sorted(current.keys()))
    raise ConfigOverrideError(
        f"Unknown environment override path segment '{lookup_key}' in {env_key}. "
        f"Allowed keys at this level: {allowed}"
    )


def _format_validation_error(error: Any) -> str:
    """Format a pydantic error for operators.

    Args:
        error: Raw pydantic error dictionary.

    Returns:
        A concise operator-facing error string.
    """
    if hasattr(error, "get"):
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "Unknown validation error")
        return f"{location}: {message}"
    return str(error)
