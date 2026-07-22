"""Generate a clearly fictional evidence package for demonstrations."""

from __future__ import annotations

from datetime import UTC, datetime

from aml_evidence_graph.evidence.package import FeatureEvidence, RiskEvidencePackage


def build_mock_evidence_package() -> RiskEvidencePackage:
    """Return synthetic data only; no private identifiers or transaction records."""
    return RiskEvidencePackage(
        alert_id="mock-alert-0001",
        generated_at=datetime(2026, 7, 22, tzinfo=UTC),
        transaction_id="mock-txn-0001",
        event_timestamp=datetime(2023, 7, 1, 9, 30, tzinfo=UTC),
        model_probabilities={"catboost": 0.42, "graphsage": 0.58},
        fusion_probability=0.55,
        key_features=[
            FeatureEvidence(
                name="sender_outgoing_count_7d",
                value=3.0,
                window="7d",
                source="mock_pit_features",
            )
        ],
        missing_evidence=["This is a fictional demo with no case narrative."],
        uncertainty_notes=["Demo scores are not production decisions."],
    )
