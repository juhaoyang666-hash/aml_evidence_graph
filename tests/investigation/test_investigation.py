from datetime import UTC, datetime
from pathlib import Path

from aml_evidence_graph.evidence.package import (
    FeatureEvidence,
    RiskEvidencePackage,
    RuleEvidence,
)
from aml_evidence_graph.evidence.typology import (
    LocalBM25TypologyRetriever,
    load_typology_documents,
)
from aml_evidence_graph.investigation.evaluation import evaluate_investigation_report
from aml_evidence_graph.investigation.workflow import run_investigation


def test_evidence_bound_investigation_preserves_facts_and_requires_review(
    tmp_path: Path,
) -> None:
    typology_path = tmp_path / "typologies"
    typology_path.mkdir()
    (typology_path / "structuring.yaml").write_text(
        """
typology_id: "TYPOLOGY-STRUCTURING"
version: "1.0"
title: "Structuring"
source: "Golden test"
body: "Repeated threshold avoidance and transaction concentration."
""".strip(),
        encoding="utf-8",
    )
    evidence = RiskEvidencePackage(
        alert_id="alert-001",
        generated_at=datetime(2026, 7, 22, tzinfo=UTC),
        transaction_id="txn-row-000000000001",
        event_timestamp=datetime(2023, 5, 1, 12, tzinfo=UTC),
        model_probabilities={"catboost": 0.2, "graphsage": 0.8},
        fusion_probability=0.7,
        rule_hits=[
            RuleEvidence(
                rule_id="R-STRUCTURING",
                rule_version="1",
                feature="sender_outgoing_count_7d",
                observed_value=8.0,
                threshold=5.0,
                operator="gte",
                explanation="Threshold reached.",
            )
        ],
        key_features=[
            FeatureEvidence(
                name="sender_outgoing_count_7d",
                value=8.0,
                window="7d",
                source="pit_features",
            )
        ],
        missing_evidence=["beneficiary purpose is not available"],
        uncertainty_notes=["Retrieved typology does not establish case intent."],
    )
    retriever = LocalBM25TypologyRetriever(load_typology_documents(typology_path))

    report = run_investigation(evidence, retriever=retriever)
    evaluation = evaluate_investigation_report(evidence, report)

    assert report.status == "draft_requires_human_review"
    assert report.fact_snapshot == evidence.model_dump(mode="json")
    assert "TYPOLOGY-STRUCTURING" in report.typology_considerations[0]
    assert evaluation.schema_valid
    assert evaluation.fact_snapshot_matches
    assert evaluation.no_evidence_refusal


def test_investigation_report_includes_evidence_bound_sar_draft() -> None:
    evidence = RiskEvidencePackage.model_validate(
        {
            "schema_version": "1.0",
            "alert_id": "mock-alert-sar-001",
            "generated_at": "2026-07-22T00:00:00Z",
            "transaction_id": "mock-txn-sar-001",
            "event_timestamp": "2023-07-01T09:30:00Z",
            "model_probabilities": {"catboost": 0.55},
            "fusion_probability": 0.53,
            "rule_hits": [],
            "key_features": [],
            "typology_references": [],
            "source_versions": {"demo": "1"},
            "missing_evidence": ["Need source-of-funds confirmation."],
            "uncertainty_notes": ["Synthetic SAR draft smoke case."],
        }
    )
    project_root = Path(__file__).resolve().parents[2]
    retriever = LocalBM25TypologyRetriever(
        load_typology_documents(project_root / "knowledge" / "typologies")
    )
    report = run_investigation(evidence, retriever=retriever)

    assert report.status == "draft_requires_human_review"
    assert report.sar_draft is not None
    assert report.sar_draft.supporting_evidence_refs
    assert "model_probabilities.catboost" in report.sar_draft.supporting_evidence_refs
    assert "Need source-of-funds confirmation." in report.sar_draft.pending_verification
    assert "regulatory filing" in report.sar_draft.disclaimer.lower()
