"""Validated data-quality configuration tied to the pre-registered time splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from aml_evidence_graph.data.splits import SPLIT_BOUNDS


def _default_data_config_path() -> Path:
    """Prefer a checked-out config, then use the config bundled in the wheel."""
    working_copy = Path.cwd() / "configs" / "data.yaml"
    if working_copy.is_file():
        return working_copy
    return Path(__file__).resolve().parents[1] / "configs" / "data.yaml"


DEFAULT_DATA_CONFIG_PATH = _default_data_config_path()


@dataclass(frozen=True)
class DataQualityPolicy:
    max_null_rate_for_required_columns: float
    require_binary_label: bool
    reject_duplicate_transaction_ids: bool


@dataclass(frozen=True)
class DataConfiguration:
    version: str
    quality: DataQualityPolicy


def _expected_split_values() -> dict[str, str]:
    return {
        f"{split.value}_start": bounds[0].isoformat()
        for split, bounds in SPLIT_BOUNDS.items()
    } | {
        f"{split.value}_end": bounds[1].isoformat()
        for split, bounds in SPLIT_BOUNDS.items()
    }


def load_data_configuration(path: Path) -> DataConfiguration:
    """Load quality gates and reject a config that silently changes the time protocol."""
    if not path.is_file():
        raise FileNotFoundError(f"Data configuration does not exist: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Data configuration must be a YAML mapping.")
    version = str(document.get("version", "")).strip()
    splits = document.get("time_splits")
    quality = document.get("quality")
    if not version or not isinstance(splits, dict) or not isinstance(quality, dict):
        raise ValueError("Data configuration requires version, time_splits, and quality.")
    expected_splits = _expected_split_values()
    mismatches = {
        name: {"expected": expected, "actual": splits.get(name)}
        for name, expected in expected_splits.items()
        if str(splits.get(name, "")) != expected
    }
    if mismatches:
        raise ValueError(
            "Data configuration may not change pre-registered time splits: "
            + ", ".join(sorted(mismatches))
        )
    try:
        maximum_null_rate = float(quality["max_null_rate_for_required_columns"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("quality.max_null_rate_for_required_columns must be numeric.") from error
    if not 0 <= maximum_null_rate <= 1:
        raise ValueError("quality.max_null_rate_for_required_columns must be in [0, 1].")
    for field_name in ("require_binary_label", "reject_duplicate_transaction_ids"):
        if not isinstance(quality.get(field_name), bool):
            raise ValueError(f"quality.{field_name} must be boolean.")
    return DataConfiguration(
        version=version,
        quality=DataQualityPolicy(
            max_null_rate_for_required_columns=maximum_null_rate,
            require_binary_label=quality["require_binary_label"],
            reject_duplicate_transaction_ids=quality["reject_duplicate_transaction_ids"],
        ),
    )
