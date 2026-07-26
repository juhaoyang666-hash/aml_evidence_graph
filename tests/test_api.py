from pathlib import Path

from fastapi.testclient import TestClient

from aml_evidence_graph.api.app import create_app
from aml_evidence_graph.evidence.typology import (
    LocalBM25TypologyRetriever,
    load_typology_documents,
)


def test_investigation_api_returns_a_review_only_draft(tmp_path: Path) -> None:
    typology_root = tmp_path / "typologies"
    typology_root.mkdir()
    (typology_root / "test.yaml").write_text(
        """
typology_id: "TYPOLOGY-TEST"
version: "1"
title: "Test Typology"
source: "Test"
body: "Transaction risk investigation."
""".strip(),
        encoding="utf-8",
    )
    retriever = LocalBM25TypologyRetriever(load_typology_documents(typology_root))
    client = TestClient(create_app(retriever))

    health = client.get("/health")
    demo = client.get("/demo")
    score_response = client.post("/v1/score/batch", json={"partition_ref": "mock-partition"})
    evidence_response = client.get("/v1/evidence/mock-alert-0001")
    response = client.post("/v1/investigations/mock-alert-0001/draft")
    demo_response = client.post("/demo/cases/mock-alert-0001/draft")
    review_response = client.post(
        "/v1/reviews",
        json={
            "alert_id": "mock-alert-0001",
            "reviewer_reference": "reviewer-test",
            "decision": "needs_more_evidence",
            "note": "Verify source records.",
        },
    )

    assert health.json() == {
        "status": "ok",
        "llm_scoring": "disabled",
        "llm_annotation": "disabled",
    }
    assert "虚构 Evidence Package" in demo.text
    assert score_response.status_code == 200
    assert evidence_response.status_code == 200
    assert response.status_code == 200
    assert response.json()["status"] == "draft_requires_human_review"
    assert demo_response.status_code == 200
    assert review_response.status_code == 200
    assert review_response.json()["model_update"] == "not_triggered"


def test_internal_v1_routes_require_token_when_configured(tmp_path: Path) -> None:
    typology_root = tmp_path / "typologies"
    typology_root.mkdir()
    (typology_root / "test.yaml").write_text(
        """
typology_id: "TYPOLOGY-TEST"
version: "1"
title: "Test Typology"
source: "Test"
body: "Transaction risk investigation."
""".strip(),
        encoding="utf-8",
    )
    retriever = LocalBM25TypologyRetriever(load_typology_documents(typology_root))
    client = TestClient(create_app(retriever, internal_api_token="internal-test-token"))

    rejected = client.post("/v1/score/batch", json={"partition_ref": "mock-partition"})
    accepted = client.post(
        "/v1/score/batch",
        json={"partition_ref": "mock-partition"},
        headers={"X-AML-Internal-Token": "internal-test-token"},
    )
    review_rejected = client.post(
        "/v1/reviews",
        json={
            "alert_id": "mock-alert-0001",
            "reviewer_reference": "reviewer-test",
            "decision": "confirmed",
        },
    )

    assert rejected.status_code == 401
    assert accepted.status_code == 200
    assert review_rejected.status_code == 401
