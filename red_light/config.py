"""
Configuration utilities for Red Light Violation Detection.

Production-ready configuration loading, validation, and path utilities to keep
scripts lean. Import from this module for all config handling.

Usage:
    from red_light.config import (
        ConfigError,
        load_training_config,
        load_evaluation_config,
        load_inference_config,
        load_data_config,
        validate_model_path,
        validate_data_yaml_path,
        validate_source_path,
    )
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import yaml


class ConfigError(Exception):
    """Raised when a configuration file is missing or invalid."""


# =============================================================================
# Core Configuration Loading
# =============================================================================

def _resolve_path(value: str, base_dir: Path) -> Path:
    """
    Resolve a potentially relative path against a base directory.
    """
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def _validate_required_fields(
    config: Dict[str, Any],
    required_fields: Iterable[str],
    config_path: Path
) -> None:
    """Validate that required top-level fields exist in config."""
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ConfigError(
            f"Missing required fields in config {config_path}: {', '.join(missing)}"
        )


def _validate_required_sections(
    config: Dict[str, Any],
    required_sections: Iterable[str],
    config_path: Path
) -> None:
    """Validate that required sections (nested dicts) exist in config."""
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(
            f"Missing required sections in config {config_path}: {', '.join(missing)}"
        )


def load_yaml_config(
    config_path: Union[str, Path],
    required_fields: Optional[Iterable[str]] = None,
    required_sections: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    """
    Load a YAML config file and optionally enforce required fields/sections.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ConfigError(f"Empty config file: {config_path}")

    if required_fields:
        _validate_required_fields(config, required_fields, config_path)

    if required_sections:
        _validate_required_sections(config, required_sections, config_path)

    return config


def _validate_file_exists(path: Path, description: str) -> None:
    """Validate that a file exists."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


# =============================================================================
# Training Configuration
# =============================================================================

# Required fields for training config
TRAINING_REQUIRED_FIELDS: List[str] = ['experiment_name', 'model', 'data_yaml']


def load_training_config(
    config_path: Union[str, Path],
    required_fields: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Load and validate a training configuration.

    Resolves relative paths (like data_yaml) based on the config file's directory.
    Validates that required files exist.
    """
    config_path = Path(config_path)

    fields = required_fields if required_fields is not None else TRAINING_REQUIRED_FIELDS
    config = load_yaml_config(config_path, required_fields=fields)

    base_dir = config_path.parent

    # Resolve and validate data_yaml path
    if "data_yaml" in config:
        resolved_data_yaml = _resolve_path(config["data_yaml"], base_dir)
        _validate_file_exists(resolved_data_yaml, "Data YAML")
        config["data_yaml"] = str(resolved_data_yaml)

    # Resolve model path if it's a local file
    if "model" in config:
        model_path = Path(config["model"])
        # Only resolve if it looks like a local path (not a model name like 'yolov8n.pt')
        if not model_path.is_absolute() and '/' in config["model"]:
            resolved_model = _resolve_path(config["model"], base_dir)
            config["model"] = str(resolved_model)

    return config


# =============================================================================
# Evaluation Configuration
# =============================================================================

# Required sections for evaluation config
EVALUATION_REQUIRED_SECTIONS: List[str] = ['evaluation', 'visualization']


def load_evaluation_config(
    config_path: Union[str, Path],
    required_sections: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Load and validate an evaluation configuration.
    """
    config_path = Path(config_path)

    sections = required_sections if required_sections is not None else EVALUATION_REQUIRED_SECTIONS
    config = load_yaml_config(config_path, required_sections=sections)

    print(f"Loaded evaluation config from: {config_path}")
    return config


# =============================================================================
# Inference Configuration
# =============================================================================

# Required sections for inference config
INFERENCE_REQUIRED_SECTIONS: List[str] = ['inference', 'visualization']


def load_inference_config(
    config_path: Union[str, Path],
    required_sections: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Load and validate an inference configuration.
    """
    config_path = Path(config_path)

    sections = required_sections if required_sections is not None else INFERENCE_REQUIRED_SECTIONS
    config = load_yaml_config(config_path, required_sections=sections)

    print(f"Loaded inference config from: {config_path}")
    return config


# =============================================================================
# Data Configuration
# =============================================================================

def load_data_config(data_yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YOLO data configuration file.
    """
    config = load_yaml_config(data_yaml_path, required_fields=['names'])
    return config


# =============================================================================
# Path Validation Utilities
# =============================================================================

def validate_model_path(model_path: Union[str, Path]) -> Path:
    """Validate that a model file exists."""
    path = Path(model_path)
    _validate_file_exists(path, "Model")
    return path


def validate_data_yaml_path(data_yaml_path: Union[str, Path]) -> Path:
    """Validate that a data YAML file exists."""
    path = Path(data_yaml_path)
    _validate_file_exists(path, "Data YAML")
    return path


def validate_source_path(source_path: Union[str, Path]) -> Path:
    """Validate that a source file or directory exists."""
    path = Path(source_path)
    _validate_file_exists(path, "Source")
    return path
