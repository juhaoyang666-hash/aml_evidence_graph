import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from aml_evidence_graph.api.app import create_app
from aml_evidence_graph.evidence.typology import (
    LocalBM25TypologyRetriever,
    load_typology_documents,
)
from aml_evidence_graph.investigation.audit_store import SQLiteInvestigationAuditStore
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


def test_controlled_api_writes_independent_minimized_audit_records(tmp_path: Path) -> None:
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
    audit_store = SQLiteInvestigationAuditStore(tmp_path / "audit.sqlite")
    client = TestClient(create_app(retriever, controlled_audit_store=audit_store))

    started = client.post(
        "/v1/controlled-investigations/mock-alert-0001",
        json={"thread_id": "audited-api-thread"},
    )
    completed = client.post(
        "/v1/controlled-investigations/audited-api-thread/review",
        json={
            "action": "edit",
            "reviewer_reference": "reviewer-audit",
            "note": "SENSITIVE REVIEW NOTE MUST NOT BE STORED",
        },
    )
    records = audit_store.list_by_thread("audited-api-thread")
    serialized = json.dumps(
        [record.model_dump(mode="json") for record in records],
        ensure_ascii=False,
    )

    assert started.status_code == 200
    assert completed.status_code == 200
    assert {record.category for record in records} == {"tool", "node", "review"}
    assert [record.name for record in records].count("human_review_decision") == 1
    assert next(record for record in records if record.category == "review").note_present
    assert "SENSITIVE REVIEW NOTE" not in serialized
    assert "feature_values" not in serialized


def test_concurrent_review_replays_execute_once_and_audit_once(tmp_path: Path) -> None:
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
    saver = create_sqlite_checkpointer(tmp_path / "checkpoint.sqlite")
    audit_store = SQLiteInvestigationAuditStore(tmp_path / "audit.sqlite")
    client = TestClient(
        create_app(
            retriever,
            controlled_checkpointer=saver,
            controlled_audit_store=audit_store,
        )
    )
    thread_id = "concurrent-idempotent-thread"
    started = client.post(
        "/v1/controlled-investigations/mock-alert-0001",
        json={"thread_id": thread_id},
    )
    request = {"action": "approve", "reviewer_reference": "reviewer-concurrent"}
    headers = {"Idempotency-Key": "review-request-0001"}

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(
                executor.map(
                    lambda _: client.post(
                        f"/v1/controlled-investigations/{thread_id}/review",
                        json=request,
                        headers=headers,
                    ),
                    range(16),
                )
            )
    finally:
        saver.conn.close()

    records = audit_store.list_by_thread(thread_id)
    assert started.status_code == 200
    assert {response.status_code for response in responses} == {200}
    assert sum(response.json()["idempotent_replay"] for response in responses) == 15
    assert {response.json()["final_status"] for response in responses} == {
        "approved_for_downstream_human_process"
    }
    assert len([record for record in records if record.category == "review"]) == 1
    assert len([record for record in records if record.name == "finalize"]) == 1


def test_review_idempotency_survives_restart_and_rejects_key_reuse(
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
    checkpoint_path = tmp_path / "checkpoint.sqlite"
    thread_id = "restart-idempotent-thread"
    headers = {"Idempotency-Key": "durable-review-request"}
    request = {"action": "reject", "reviewer_reference": "reviewer-restart"}

    first_saver = create_sqlite_checkpointer(checkpoint_path)
    first_client = TestClient(create_app(retriever, controlled_checkpointer=first_saver))
    first_client.post(
        "/v1/controlled-investigations/mock-alert-0001",
        json={"thread_id": thread_id},
    )
    completed = first_client.post(
        f"/v1/controlled-investigations/{thread_id}/review",
        json=request,
        headers=headers,
    )
    first_saver.conn.close()

    second_saver = create_sqlite_checkpointer(checkpoint_path)
    try:
        second_client = TestClient(create_app(retriever, controlled_checkpointer=second_saver))
        replayed = second_client.post(
            f"/v1/controlled-investigations/{thread_id}/review",
            json=request,
            headers=headers,
        )
        conflicting = second_client.post(
            f"/v1/controlled-investigations/{thread_id}/review",
            json={"action": "approve", "reviewer_reference": "reviewer-restart"},
            headers=headers,
        )
    finally:
        second_saver.conn.close()

    assert completed.status_code == 200
    assert not completed.json()["idempotent_replay"]
    assert replayed.status_code == 200
    assert replayed.json()["idempotent_replay"]
    assert conflicting.status_code == 409
    assert "different review request" in conflicting.json()["detail"]
