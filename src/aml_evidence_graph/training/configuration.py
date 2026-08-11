"""Validated model configuration loading shared by training and OOF commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _default_config_path(filename: str) -> Path:
    """Prefer a checked-out config, then use the config bundled in the wheel."""
    working_copy = Path.cwd() / "configs" / filename
    if working_copy.is_file():
        return working_copy
    return Path(__file__).resolve().parents[1] / "configs" / filename


DEFAULT_MODEL_CONFIG_PATH = _default_config_path("models.yaml")


def load_model_configuration(path: Path) -> dict[str, Any]:
    """Load the versioned model configuration without silently accepting malformed YAML."""
    if not path.is_file():
        raise FileNotFoundError(f"Model configuration does not exist: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Model configuration must be a YAML mapping.")
    required_sections = {"version", "catboost", "lightgbm", "graphsage", "fusion"}
    missing = sorted(required_sections.difference(document))
    if missing:
        raise ValueError("Model configuration is missing sections: " + ", ".join(missing))
    for section in required_sections:
        if section == "version":
            continue
        if not isinstance(document[section], dict):
            raise ValueError(f"Model configuration section {section} must be a mapping.")
    if not str(document["version"]).strip():
        raise ValueError("Model configuration version must be non-empty.")
    return document


def catboost_parameters_from_configuration(document: dict[str, Any]) -> dict[str, Any]:
    """Return CatBoost parameters while leaving the run seed under CLI control."""
    parameters = dict(document["catboost"])
    parameters.pop("random_seed", None)
    return parameters


def lightgbm_parameters_from_configuration(document: dict[str, Any]) -> dict[str, Any]:
    """Return promoted LightGBM parameters while keeping weights and seed centralized."""
    parameters = dict(document["lightgbm"])
    for name in ("random_state", "random_seed", "scale_pos_weight", "class_weight"):
        parameters.pop(name, None)
    return parameters


def graphsage_parameters_from_configuration(document: dict[str, Any]) -> dict[str, Any]:
    """Map the public YAML contract to GraphSAGETrainingConfig keyword names."""
    source = document["graphsage"]
    allowed = {
        "architecture",
        "hidden_dim",
        "num_layers",
        "num_neighbors",
        "batch_size",
        "epochs",
        "learning_rate",
        "dropout",
        "history_window_days",
        "num_relations",
    }
    documented_fields = {"early_stopping_patience"}
    unknown = sorted(set(source).difference(allowed | documented_fields))
    if unknown:
        raise ValueError("Unsupported GraphSAGE configuration fields: " + ", ".join(unknown))
    parameters = {name: source[name] for name in allowed if name in source}
    if "num_neighbors" in parameters:
        parameters["num_neighbors"] = tuple(int(value) for value in parameters["num_neighbors"])
    if "early_stopping_patience" in source:
        parameters["patience"] = int(source["early_stopping_patience"])
    return parameters


def fusion_alert_fraction_from_configuration(document: dict[str, Any]) -> float:
    """Return the pre-registered validation alert budget fraction."""
    value = float(document["fusion"].get("alert_budget_fraction", 0.005))
    if not 0 < value <= 1:
        raise ValueError("fusion.alert_budget_fraction must be in (0, 1].")
    return value
