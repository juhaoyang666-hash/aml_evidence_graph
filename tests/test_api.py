from pathlib import Path

from fastapi.testclient import TestClient

from aml_evidence_graph.api.app import create_app
from aml_evidence_graph.evidence.typology import (
    LocalBM25TypologyRetriever,
    load_typology_documents,
)
from aml_evidence_graph.investigation.workflow_v2 import create_sqlite_checkpointer


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


def test_controlled_investigation_api_pauses_queries_and_resumes(tmp_path: Path) -> None:
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

    started = client.post(
        "/v1/controlled-investigations/mock-alert-0001",
        json={"thread_id": "api-controlled-1"},
    )
    queried = client.get("/v1/controlled-investigations/api-controlled-1")
    duplicate = client.post(
        "/v1/controlled-investigations/mock-alert-0001",
        json={"thread_id": "api-controlled-1"},
    )
    completed = client.post(
        "/v1/controlled-investigations/api-controlled-1/review",
        json={"action": "approve", "reviewer_reference": "reviewer-api"},
    )
    repeated_review = client.post(
        "/v1/controlled-investigations/api-controlled-1/review",
        json={"action": "reject", "reviewer_reference": "reviewer-api"},
    )

    assert started.status_code == 200
    assert started.json()["status"] == "awaiting_human_review"
    assert started.json()["review_prompt"]["kind"] == "human_review"
    assert started.json()["report"]["status"] == "draft_requires_human_review"
    assert queried.json()["status"] == "awaiting_human_review"
    assert duplicate.status_code == 409
    assert completed.json()["status"] == "completed"
    assert completed.json()["final_status"] == "approved_for_downstream_human_process"
    assert repeated_review.status_code == 409


def test_controlled_investigation_routes_share_internal_authentication(tmp_path: Path) -> None:
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

    rejected = client.post(
        "/v1/controlled-investigations/mock-alert-0001",
        json={"thread_id": "protected-thread"},
    )
    accepted = client.post(
        "/v1/controlled-investigations/mock-alert-0001",
        json={"thread_id": "protected-thread"},
        headers={"X-AML-Internal-Token": "internal-test-token"},
    )

    assert rejected.status_code == 401
    assert accepted.status_code == 200


def test_controlled_investigation_api_resumes_from_sqlite_after_restart(
    tmp_path: Path,
) -> None:
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
    checkpoint_path = tmp_path / "controlled.sqlite"

    first_saver = create_sqlite_checkpointer(checkpoint_path)
    first_client = TestClient(create_app(retriever, controlled_checkpointer=first_saver))
    started = first_client.post(
        "/v1/controlled-investigations/mock-alert-0001",
        json={"thread_id": "persistent-api-thread"},
    )
    assert started.status_code == 200
    assert started.json()["status"] == "awaiting_human_review"
    first_saver.conn.close()

    second_saver = create_sqlite_checkpointer(checkpoint_path)
    second_client = TestClient(create_app(retriever, controlled_checkpointer=second_saver))
    restored = second_client.get(
        "/v1/controlled-investigations/persistent-api-thread"
    )
    completed = second_client.post(
        "/v1/controlled-investigations/persistent-api-thread/review",
        json={"action": "approve", "reviewer_reference": "reviewer-restart"},
    )
    second_saver.conn.close()

    assert restored.status_code == 200
    assert restored.json()["status"] == "awaiting_human_review"
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
