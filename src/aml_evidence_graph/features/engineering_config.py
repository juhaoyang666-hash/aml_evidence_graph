"""Versioned constants for typology-proxy and amount/geography feature flags."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _default_engineering_config_path() -> Path:
    working_copy = Path.cwd() / "configs" / "feature_engineering.yaml"
    if working_copy.is_file():
        return working_copy
    return Path(__file__).resolve().parents[1] / "configs" / "feature_engineering.yaml"


DEFAULT_FEATURE_ENGINEERING_CONFIG_PATH = _default_engineering_config_path()


@dataclass(frozen=True, slots=True)
class FeatureEngineeringConfig:
    """Auditable constants consumed by ``PITFeatureBuilder`` and registry metadata."""

    version: str
    high_risk_locations: frozenset[str]
    cash_like_payment_types: frozenset[str]
    cross_border_payment_types: frozenset[str]
    reporting_threshold: float
    just_below_reporting_threshold_ratio: float
    round_amount_tolerance: float
    small_amount_threshold: float
    cash_in_then_out_window_hours: float
    missing_recency_seconds: float

    @classmethod
    def defaults(cls) -> FeatureEngineeringConfig:
        """Conservative built-in defaults mirroring ``configs/feature_engineering.yaml``."""
        return cls(
            version="fe-v2-defaults",
            high_risk_locations=frozenset(
                {
                    "Mexico",
                    "Turkey",
                    "Morocco",
                    "UAE",
                    "United Arab Emirates",
                    "Nigeria",
                    "Iran",
                    "Myanmar",
                }
            ),
            cash_like_payment_types=frozenset(
                {
                    "Cash",
                    "Cash Deposit",
                    "Cash Withdrawal",
                    "CASH",
                    "CASH DEPOSIT",
                    "CASH WITHDRAWAL",
                }
            ),
            cross_border_payment_types=frozenset(
                {
                    "Cross-border",
                    "Cross Border",
                    "CROSS-BORDER",
                    "CROSS_BORDER",
                }
            ),
            reporting_threshold=45225.28,
            just_below_reporting_threshold_ratio=0.90,
            round_amount_tolerance=0.01,
            small_amount_threshold=1000.0,
            cash_in_then_out_window_hours=24.0,
            missing_recency_seconds=2_592_000.0,
        )


def _require_positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return float(value)


def _require_unit_interval(name: str, value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1).")
    return float(value)


def _string_set(document: dict[str, Any], key: str) -> frozenset[str]:
    raw = document.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Feature engineering config requires a non-empty list for {key}.")
    values = {str(item).strip() for item in raw if str(item).strip()}
    if not values:
        raise ValueError(f"Feature engineering config list {key} contains no usable values.")
    return frozenset(values)


def load_feature_engineering_config(path: Path) -> FeatureEngineeringConfig:
    """Load typology-proxy constants from a versioned YAML file."""
    if not path.is_file():
        raise FileNotFoundError(f"Feature engineering configuration does not exist: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Feature engineering configuration must be a YAML mapping.")
    version = str(document.get("version", "")).strip()
    if not version:
        raise ValueError("Feature engineering configuration requires version.")
    return FeatureEngineeringConfig(
        version=version,
        high_risk_locations=_string_set(document, "high_risk_locations"),
        cash_like_payment_types=_string_set(document, "cash_like_payment_types"),
        cross_border_payment_types=_string_set(document, "cross_border_payment_types"),
        reporting_threshold=_require_positive(
            "reporting_threshold",
            float(document["reporting_threshold"]),
        ),
        just_below_reporting_threshold_ratio=_require_unit_interval(
            "just_below_reporting_threshold_ratio",
            float(document["just_below_reporting_threshold_ratio"]),
        ),
        round_amount_tolerance=_require_positive(
            "round_amount_tolerance",
            float(document["round_amount_tolerance"]),
        ),
        small_amount_threshold=_require_positive(
            "small_amount_threshold",
            float(document["small_amount_threshold"]),
        ),
        cash_in_then_out_window_hours=_require_positive(
            "cash_in_then_out_window_hours",
            float(document["cash_in_then_out_window_hours"]),
        ),
        missing_recency_seconds=_require_positive(
            "missing_recency_seconds",
            float(document["missing_recency_seconds"]),
        ),
    )
