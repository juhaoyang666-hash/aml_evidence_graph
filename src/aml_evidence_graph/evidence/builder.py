"""Deterministically construct RiskEvidencePackage objects from private artifacts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

import polars as pl

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.evidence.package import (
    FeatureEvidence,
    GraphEvidence,
    RiskEvidencePackage,
    RuleEvidence,
)
from aml_evidence_graph.graph.explain import GraphEdgeEvidence
from aml_evidence_graph.rules.engine import RuleHit


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _as_row_mapping(scored_transaction: Mapping[str, object] | pl.Series | object) -> Mapping[str, object]:
    if isinstance(scored_transaction, Mapping):
        return scored_transaction
    if isinstance(scored_transaction, pl.Series):
        frame = scored_transaction.to_frame()
        return dict(zip(frame.columns, scored_transaction.to_list(), strict=True))
    if hasattr(scored_transaction, "to_dict"):
        return scored_transaction.to_dict()
    raise TypeError(f"Unsupported scored transaction type: {type(scored_transaction)!r}")


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = pl.Series([value]).cast(pl.Datetime(time_zone="UTC"), strict=True).item()
        if not isinstance(parsed, datetime):
            raise ValueError("event_ts could not be parsed as a UTC datetime.")
        timestamp = parsed
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _rule_evidence(hits: Iterable[RuleHit]) -> list[RuleEvidence]:
    return [
        RuleEvidence(
            rule_id=hit.rule_id,
            rule_version=hit.rule_version,
            feature=hit.feature,
            observed_value=hit.observed_value,
            threshold=hit.threshold,
            operator=hit.operator,
            explanation=hit.explanation,
        )
        for hit in hits
    ]


def _graph_evidence(
    evidence: GraphEdgeEvidence | None,
    *,
    snapshot_as_of: datetime | None,
) -> GraphEvidence | None:
    if evidence is None:
        return None
    if snapshot_as_of is None:
        raise ValueError("snapshot_as_of is required when graph evidence is supplied.")
    return GraphEvidence(
        source_node_index=evidence.source_node,
        destination_node_index=evidence.destination_node,
        historical_source_out_degree=evidence.historical_source_out_degree,
        historical_destination_in_degree=evidence.historical_destination_in_degree,
        prior_directed_edge_count=evidence.prior_directed_edge_count,
        prior_reverse_edge_count=evidence.prior_reverse_edge_count,
        two_hop_intermediary_count=len(evidence.two_hop_intermediary_nodes),
        two_hop_intermediary_node_indices=list(evidence.two_hop_intermediary_nodes),
        snapshot_as_of=snapshot_as_of,
        interpretation_limit=evidence.interpretation_limit,
    )


def build_risk_evidence_package(
    scored_transaction: Mapping[str, object] | pl.Series | dict[str, object],
    *,
    alert_id: str,
    model_probabilities: dict[str, float],
    source_versions: dict[str, str],
    selected_feature_names: Iterable[str],
    rule_hits: Iterable[RuleHit] = (),
    graph_edge_evidence: GraphEdgeEvidence | None = None,
    graph_snapshot_as_of: datetime | None = None,
    fusion_probability: float | None = None,
    missing_evidence: Iterable[str] = (),
    uncertainty_notes: Iterable[str] = (),
) -> RiskEvidencePackage:
    """Construct facts solely from one scored transaction and approved artifacts."""
    row = _as_row_mapping(scored_transaction)
    required = {CANONICAL.transaction_id, CANONICAL.event_ts}
    missing = sorted(required.difference(row.keys()))
    if missing:
        raise ValueError(f"Scored transaction is missing: {', '.join(missing)}")
    features: list[FeatureEvidence] = []
    for name in selected_feature_names:
        if name not in row:
            raise ValueError(f"Selected evidence feature is absent: {name}")
        value = row[name]
        normalized: float | str | bool | None
        if _is_missing(value):
            normalized = None
        elif isinstance(value, bool):
            normalized = value
        elif isinstance(value, (int, float)):
            normalized = float(value)
        else:
            normalized = str(value)
        features.append(
            FeatureEvidence(
                name=name,
                value=normalized,
                source="pit_feature_dataset",
            )
        )
    return RiskEvidencePackage(
        alert_id=alert_id,
        generated_at=datetime.now(UTC),
        transaction_id=str(row[CANONICAL.transaction_id]),
        event_timestamp=_as_utc_datetime(row[CANONICAL.event_ts]),
        model_probabilities=model_probabilities,
        fusion_probability=fusion_probability,
        rule_hits=_rule_evidence(rule_hits),
        key_features=features,
        graph_evidence=_graph_evidence(
            graph_edge_evidence,
            snapshot_as_of=graph_snapshot_as_of,
        ),
        source_versions=source_versions,
        missing_evidence=list(missing_evidence),
        uncertainty_notes=list(uncertainty_notes),
    )
