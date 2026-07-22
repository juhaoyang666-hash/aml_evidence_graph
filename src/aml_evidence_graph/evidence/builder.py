"""Deterministically construct RiskEvidencePackage objects from private artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

import pandas as pd

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.evidence.package import (
    FeatureEvidence,
    GraphEvidence,
    RiskEvidencePackage,
    RuleEvidence,
)
from aml_evidence_graph.graph.explain import GraphEdgeEvidence
from aml_evidence_graph.rules.engine import RuleHit


def _as_utc_datetime(value: object) -> datetime:
    timestamp = pd.to_datetime(value, utc=True, errors="raise")
    return timestamp.to_pydatetime().astimezone(UTC)


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
    scored_transaction: pd.Series,
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
    required = {CANONICAL.transaction_id, CANONICAL.event_ts}
    missing = sorted(required.difference(scored_transaction.index))
    if missing:
        raise ValueError(f"Scored transaction is missing: {', '.join(missing)}")
    features: list[FeatureEvidence] = []
    for name in selected_feature_names:
        if name not in scored_transaction:
            raise ValueError(f"Selected evidence feature is absent: {name}")
        value = scored_transaction[name]
        normalized: float | str | bool | None
        if pd.isna(value):
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
        transaction_id=str(scored_transaction[CANONICAL.transaction_id]),
        event_timestamp=_as_utc_datetime(scored_transaction[CANONICAL.event_ts]),
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
