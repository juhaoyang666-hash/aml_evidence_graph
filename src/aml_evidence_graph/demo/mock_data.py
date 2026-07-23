"""Generate a clearly fictional evidence package for demonstrations."""

from __future__ import annotations

from datetime import UTC, datetime

from aml_evidence_graph.evidence.package import (
    FeatureEvidence,
    RiskEvidencePackage,
    TypologyReference,
)


def build_mock_evidence_package() -> RiskEvidencePackage:
    """Return synthetic data only; no private identifiers or transaction records."""
    return RiskEvidencePackage(
        alert_id="mock-alert-0001",
        generated_at=datetime(2026, 7, 22, tzinfo=UTC),
        transaction_id="mock-txn-0001",
        event_timestamp=datetime(2023, 7, 1, 9, 30, tzinfo=UTC),
        model_probabilities={
            "catboost": 0.42,
            "graph_stats_catboost": 0.48,
            "graphsage": 0.58,
        },
        fusion_probability=0.55,
        key_features=[
            FeatureEvidence(
                name="sender_outgoing_count_7d",
                value=3.0,
                window="7d",
                source="mock_pit_features",
            ),
            FeatureEvidence(
                name="graph_directed_edge_prior_count",
                value=2.0,
                window="historical_pre_event",
                source="mock_graph_stats",
            ),
        ],
        typology_references=[
            TypologyReference(
                typology_id="TYPOLOGY-STRUCTURING",
                version="2026.1",
                title="Structuring / Smurfing",
                source="demo-seed",
            )
        ],
        source_versions={"demo": "1", "dataset_framing": "synthetic-SAML-D-demo-only"},
        missing_evidence=[
            "Fictional demo package; no underlying SAML-D transaction payload is loaded.",
            "No investigator notes or source-of-funds documents are available in Demo mode.",
        ],
        uncertainty_notes=[
            "Demo scores are illustrative placeholders, not frozen-model outputs.",
            "Do not cite Demo probabilities as SAML-D evaluation metrics.",
        ],
    )
