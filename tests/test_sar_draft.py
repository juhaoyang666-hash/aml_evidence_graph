from pathlib import Path

from aml_evidence_graph.evidence.package import RiskEvidencePackage
from aml_evidence_graph.evidence.typology import LocalBM25TypologyRetriever, load_typology_documents
from aml_evidence_graph.investigation.workflow import run_investigation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    retriever = LocalBM25TypologyRetriever(
        load_typology_documents(PROJECT_ROOT / "knowledge" / "typologies")
    )
    report = run_investigation(evidence, retriever=retriever)

    assert report.status == "draft_requires_human_review"
    assert report.sar_draft is not None
    assert report.sar_draft.supporting_evidence_refs
    assert "model_probabilities.catboost" in report.sar_draft.supporting_evidence_refs
    assert "Need source-of-funds confirmation." in report.sar_draft.pending_verification
    assert "regulatory filing" in report.sar_draft.disclaimer.lower()
