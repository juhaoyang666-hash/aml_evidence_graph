from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aml_evidence_graph.api.app import create_app
from aml_evidence_graph.api.services import SQLiteEvidenceStore
from aml_evidence_graph.demo.mock_data import build_mock_evidence_package
from aml_evidence_graph.evidence.typology import (
    LocalBM25TypologyRetriever,
    load_typology_documents,
)
from aml_evidence_graph.investigation.audit_store import SQLiteInvestigationAuditStore
from aml_evidence_graph.investigation.coordination import SQLiteThreadLockRegistry
from aml_evidence_graph.investigation.workflow_v2 import create_sqlite_checkpointer


def _retriever(tmp_path: Path) -> LocalBM25TypologyRetriever:
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
    return LocalBM25TypologyRetriever(load_typology_documents(typology_root))


def test_sqlite_evidence_is_visible_to_another_store_and_api_worker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.sqlite"
    writer = SQLiteEvidenceStore(path)
    reader = SQLiteEvidenceStore(path)
    evidence = build_mock_evidence_package().model_copy(
        update={"alert_id": "cross-worker-alert"}
    )
    writer.put(evidence)

    client = TestClient(create_app(_retriever(tmp_path), evidence_store=reader))
    response = client.get("/v1/evidence/cross-worker-alert")

    assert reader.get("cross-worker-alert") == evidence
    assert response.status_code == 200
    assert response.json()["alert_id"] == "cross-worker-alert"

    conflicting = evidence.model_copy(update={"fusion_probability": 0.123})
    with pytest.raises(ValueError, match="different immutable payload"):
        writer.put(conflicting)


def test_two_api_workers_share_thread_lock_checkpoint_and_exactly_once_audit(
    tmp_path: Path,
) -> None:
    retriever = _retriever(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.sqlite"
    audit_path = tmp_path / "audit.sqlite"
    coordination_path = tmp_path / "coordination.sqlite"
    evidence_path = tmp_path / "evidence.sqlite"
    saver_one = create_sqlite_checkpointer(checkpoint_path)
    saver_two = create_sqlite_checkpointer(checkpoint_path)
    client_one = TestClient(
        create_app(
            retriever,
            evidence_store=SQLiteEvidenceStore(evidence_path),
            controlled_checkpointer=saver_one,
            controlled_audit_store=SQLiteInvestigationAuditStore(audit_path),
            thread_lock_registry=SQLiteThreadLockRegistry(coordination_path),
        )
    )
    client_two = TestClient(
        create_app(
            retriever,
            evidence_store=SQLiteEvidenceStore(evidence_path),
            controlled_checkpointer=saver_two,
            controlled_audit_store=SQLiteInvestigationAuditStore(audit_path),
            thread_lock_registry=SQLiteThreadLockRegistry(coordination_path),
        )
    )
    thread_id = "shared-worker-thread"
    started = client_one.post(
        "/v1/controlled-investigations/mock-alert-0001",
        json={"thread_id": thread_id},
    )
    restored = client_two.get(f"/v1/controlled-investigations/{thread_id}")
    request = {"action": "approve", "reviewer_reference": "shared-reviewer"}
    headers = {"Idempotency-Key": "shared-review-key"}

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(
                executor.map(
                    lambda client: client.post(
                        f"/v1/controlled-investigations/{thread_id}/review",
                        json=request,
                        headers=headers,
                    ),
                    (client_one, client_two),
                )
            )
    finally:
        saver_one.conn.close()
        saver_two.conn.close()

    records = SQLiteInvestigationAuditStore(audit_path).list_by_thread(thread_id)
    serialized = json.dumps(
        [record.model_dump(mode="json") for record in records], sort_keys=True
    )
    assert started.status_code == 200
    assert restored.status_code == 200
    assert {response.status_code for response in responses} == {200}
    assert sum(response.json()["idempotent_replay"] for response in responses) == 1
    assert len([record for record in records if record.category == "review"]) == 1
    assert len([record for record in records if record.name == "finalize"]) == 1
    assert serialized.count("human_review_decision") == 1
