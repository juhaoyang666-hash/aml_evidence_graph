"""Versioned, machine-readable metadata for generated AML feature columns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.features.pit import WINDOWS
from aml_evidence_graph.rules.engine import RuleDefinition


def _default_registry_path() -> Path:
    """Prefer a checked-out registry, then use the config bundled in the wheel."""
    working_copy = Path.cwd() / "configs" / "features.yaml"
    if working_copy.is_file():
        return working_copy
    return Path(__file__).resolve().parents[1] / "configs" / "features.yaml"


DEFAULT_FEATURE_REGISTRY_PATH = _default_registry_path()


@dataclass(frozen=True)
class FeatureMetadata:
    """Documentation needed to assess whether one model feature is usable."""

    feature_name: str
    owner: str
    source_columns: tuple[str, ...]
    window: str
    available_at: str
    version: str
    unit_test: str


def _load_registry_configuration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature registry configuration does not exist: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Feature registry configuration must be a YAML mapping.")
    version = str(document.get("version", "")).strip()
    owners = document.get("owners")
    if not version or not isinstance(owners, dict):
        raise ValueError("Feature registry configuration requires version and owners.")
    required_owners = {
        "current_transaction",
        "point_in_time_history",
        "causal_graph_statistics",
        "typology_proxies",
        "approved_rules",
    }
    missing = sorted(required_owners.difference(owners))
    if missing or any(not str(owners[name]).strip() for name in required_owners if name in owners):
        raise ValueError("Feature registry has invalid owners: " + ", ".join(missing))
    return {"version": version, "owners": {name: str(value) for name, value in owners.items()}}


def load_static_feature_metadata(path: Path) -> list[FeatureMetadata]:
    """Materialize metadata for every non-rule feature emitted by current builders."""
    configuration = _load_registry_configuration(path)
    version = configuration["version"]
    owners: dict[str, str] = configuration["owners"]
    result: list[FeatureMetadata] = []

    def add(
        feature_name: str,
        *,
        owner_group: str,
        source_columns: tuple[str, ...],
        window: str,
        available_at: str,
        unit_test: str,
    ) -> None:
        result.append(
            FeatureMetadata(
                feature_name=feature_name,
                owner=owners[owner_group],
                source_columns=source_columns,
                window=window,
                available_at=available_at,
                version=version,
                unit_test=unit_test,
            )
        )

    add(
        "is_new_sender_account",
        owner_group="current_transaction",
        source_columns=(CANONICAL.sender_account_id, CANONICAL.event_ts),
        window="current_transaction",
        available_at="event_time",
        unit_test="tests/test_pit_features.py",
    )
    add(
        "is_new_receiver_account",
        owner_group="current_transaction",
        source_columns=(CANONICAL.receiver_account_id, CANONICAL.event_ts),
        window="current_transaction",
        available_at="event_time",
        unit_test="tests/test_pit_features.py",
    )
    add(
        "is_cross_border_current_transaction",
        owner_group="current_transaction",
        source_columns=(CANONICAL.sender_location, CANONICAL.receiver_location),
        window="current_transaction",
        available_at="event_time",
        unit_test="tests/test_pit_features.py",
    )
    add(
        "amount_log1p",
        owner_group="current_transaction",
        source_columns=(CANONICAL.amount,),
        window="current_transaction",
        available_at="event_time",
        unit_test="tests/test_pit_features.py",
    )
    add(
        "is_currency_conversion",
        owner_group="current_transaction",
        source_columns=(CANONICAL.payment_currency, CANONICAL.received_currency),
        window="current_transaction",
        available_at="event_time",
        unit_test="tests/test_pit_features.py",
    )

    typology_current = (
        (
            "is_high_risk_sender_location",
            (CANONICAL.sender_location,),
            "current_transaction",
            "event_time",
        ),
        (
            "is_high_risk_receiver_location",
            (CANONICAL.receiver_location,),
            "current_transaction",
            "event_time",
        ),
        (
            "is_high_risk_corridor",
            (CANONICAL.sender_location, CANONICAL.receiver_location),
            "current_transaction",
            "event_time",
        ),
        (
            "is_cash_like_payment",
            (CANONICAL.payment_type,),
            "current_transaction",
            "event_time",
        ),
        (
            "is_cross_border_payment_type",
            (CANONICAL.payment_type,),
            "current_transaction",
            "event_time",
        ),
        (
            "is_round_amount",
            (CANONICAL.amount,),
            "current_transaction",
            "event_time",
        ),
        (
            "is_just_below_reporting_threshold",
            (CANONICAL.amount,),
            "current_transaction",
            "event_time",
        ),
        (
            "hour_of_day",
            (CANONICAL.event_ts,),
            "current_transaction",
            "event_time",
        ),
        (
            "day_of_week",
            (CANONICAL.event_ts,),
            "current_transaction",
            "event_time",
        ),
        (
            "is_weekend",
            (CANONICAL.event_ts,),
            "current_transaction",
            "event_time",
        ),
    )
    for feature_name, source_columns, window, available_at in typology_current:
        add(
            feature_name,
            owner_group="typology_proxies",
            source_columns=source_columns,
            window=window,
            available_at=available_at,
            unit_test="tests/test_pit_features.py",
        )

    typology_history = (
        (
            "sender_small_amount_unique_receivers_7d",
            (CANONICAL.sender_account_id, CANONICAL.receiver_account_id, CANONICAL.amount),
            "[7d, event_time)",
        ),
        (
            "receiver_small_amount_unique_senders_7d",
            (CANONICAL.receiver_account_id, CANONICAL.sender_account_id, CANONICAL.amount),
            "[7d, event_time)",
        ),
        (
            "amount_to_sender_outgoing_mean_ratio_30d",
            (CANONICAL.sender_account_id, CANONICAL.amount, CANONICAL.event_ts),
            "[30d, event_time)",
        ),
        (
            "amount_zscore_vs_sender_outgoing_30d",
            (CANONICAL.sender_account_id, CANONICAL.amount, CANONICAL.event_ts),
            "[30d, event_time)",
        ),
        (
            "seconds_since_last_outgoing",
            (CANONICAL.sender_account_id, CANONICAL.event_ts),
            "prior_sender_outgoing",
        ),
        (
            "seconds_since_last_incoming",
            (CANONICAL.sender_account_id, CANONICAL.event_ts),
            "prior_sender_incoming",
        ),
        (
            "cash_in_then_out_within_window",
            (CANONICAL.sender_account_id, CANONICAL.payment_type, CANONICAL.event_ts),
            "configured_cash_in_then_out_window",
        ),
        (
            "sender_outgoing_count_1d_over_30d",
            (CANONICAL.sender_account_id, CANONICAL.event_ts),
            "[1d,30d] ratio",
        ),
        (
            "receiver_incoming_count_1d_over_30d",
            (CANONICAL.receiver_account_id, CANONICAL.event_ts),
            "[1d,30d] ratio",
        ),
        (
            "sender_outgoing_unique_counterparties_1d_over_30d",
            (CANONICAL.sender_account_id, CANONICAL.event_ts),
            "[1d,30d] ratio",
        ),
        (
            "receiver_incoming_unique_counterparties_1d_over_30d",
            (CANONICAL.receiver_account_id, CANONICAL.event_ts),
            "[1d,30d] ratio",
        ),
        (
            "any_rule_hit",
            ("rule_*_hit",),
            "after_rule_evaluation",
        ),
        (
            "rule_hit_count",
            ("rule_*_hit",),
            "after_rule_evaluation",
        ),
    )
    for feature_name, source_columns, window in typology_history:
        available_at = (
            "after_rule_evaluation"
            if feature_name in {"any_rule_hit", "rule_hit_count"}
            else "strictly_before_event_time"
        )
        add(
            feature_name,
            owner_group="typology_proxies",
            source_columns=source_columns,
            window=window,
            available_at=available_at,
            unit_test="tests/test_pit_features.py",
        )

    history_definitions = (
        ("count", (CANONICAL.event_ts,)),
        ("same_currency_amount_sum", (CANONICAL.event_ts, CANONICAL.amount)),
        ("unique_counterparties", (CANONICAL.event_ts,)),
        (
            "cross_border_count",
            (
                CANONICAL.event_ts,
                CANONICAL.sender_location,
                CANONICAL.receiver_location,
            ),
        ),
    )
    for prefix, account_column, counterparty_column in (
        ("sender_outgoing", CANONICAL.sender_account_id, CANONICAL.receiver_account_id),
        ("receiver_incoming", CANONICAL.receiver_account_id, CANONICAL.sender_account_id),
        ("relationship", CANONICAL.sender_account_id, CANONICAL.receiver_account_id),
    ):
        for window in WINDOWS:
            for suffix, extra_sources in history_definitions:
                add(
                    f"{prefix}_{suffix}_{window}",
                    owner_group="point_in_time_history",
                    source_columns=(account_column, counterparty_column, *extra_sources),
                    window=f"[{window}, event_time)",
                    available_at="strictly_before_event_time",
                    unit_test="tests/test_pit_features.py",
                )

    graph_definitions = (
        ("graph_sender_historical_out_degree", (CANONICAL.sender_account_id,)),
        ("graph_sender_historical_in_degree", (CANONICAL.sender_account_id,)),
        ("graph_receiver_historical_out_degree", (CANONICAL.receiver_account_id,)),
        ("graph_receiver_historical_in_degree", (CANONICAL.receiver_account_id,)),
        (
            "graph_directed_edge_prior_count",
            (CANONICAL.sender_account_id, CANONICAL.receiver_account_id),
        ),
        (
            "graph_reverse_edge_prior_count",
            (CANONICAL.sender_account_id, CANONICAL.receiver_account_id),
        ),
        (
            "graph_prior_reciprocal_relationship",
            (CANONICAL.sender_account_id, CANONICAL.receiver_account_id),
        ),
    )
    for feature_name, source_columns in graph_definitions:
        add(
            feature_name,
            owner_group="causal_graph_statistics",
            source_columns=source_columns,
            window="all_strictly_prior_events",
            available_at="strictly_before_event_time",
            unit_test="tests/test_graph_stats.py",
        )
    return result


def rule_feature_metadata(
    rules: list[RuleDefinition],
    *,
    path: Path,
) -> list[FeatureMetadata]:
    """Document active rule features without treating rules as labels."""
    configuration = _load_registry_configuration(path)
    return [
        FeatureMetadata(
            feature_name=f"rule_{rule.rule_id}_hit",
            owner=configuration["owners"]["approved_rules"],
            source_columns=rule.required_features,
            window="rule_defined_pit_window",
            available_at="after_pit_feature_calculation",
            version=rule.version,
            unit_test="tests/test_rules.py",
        )
        for rule in rules
        if rule.active
    ]


def validate_feature_metadata(
    feature_columns: set[str],
    metadata: list[FeatureMetadata],
) -> None:
    """Fail a build when a generated model feature has no auditable definition."""
    names = [item.feature_name for item in metadata]
    if len(names) != len(set(names)):
        raise ValueError("Feature registry contains duplicate feature names.")
    missing = sorted(feature_columns.difference(names))
    if missing:
        raise ValueError("Generated features missing registry metadata: " + ", ".join(missing))


def write_feature_registry(output_path: Path, metadata: list[FeatureMetadata]) -> None:
    """Persist a non-sensitive feature contract beside a private feature dataset."""
    payload = {
        "registry_schema_version": "1.0",
        "feature_count": len(metadata),
        "features": [asdict(item) for item in sorted(metadata, key=lambda item: item.feature_name)],
    }
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
