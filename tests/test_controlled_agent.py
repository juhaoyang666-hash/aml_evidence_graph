from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from aml_evidence_graph.demo.mock_data import build_mock_evidence_package
from aml_evidence_graph.evidence.typology import LocalBM25TypologyRetriever, TypologyDocument
from aml_evidence_graph.investigation.tools import (
    BoundedSubgraphInput,
    InvestigationToolCall,
    InvestigationToolRegistry,
)
from aml_evidence_graph.investigation.workflow_v2 import (
    HumanReviewDecision,
    build_controlled_investigation_graph,
    create_sqlite_checkpointer,
    resume_controlled_investigation,
    start_controlled_investigation,
)


def _retriever() -> LocalBM25TypologyRetriever:
    return LocalBM25TypologyRetriever(
        [
            TypologyDocument(
                typology_id="structuring",
                version="1",
                title="Structuring",
                body="repeated split transfers",
                source="test",
            )
        ]
    )


def test_controlled_agent_pauses_and_resumes_after_review() -> None:
    evidence = build_mock_evidence_package()
    retriever = _retriever()
    graph = build_controlled_investigation_graph(
        InvestigationToolRegistry(retriever),
        retriever=retriever,
        checkpointer=InMemorySaver(),
    )

    paused = start_controlled_investigation(graph, evidence, thread_id="case-1")

    assert paused["__interrupt__"]
    assert paused["report"]["status"] == "draft_requires_human_review"
    assert len(paused["audit_events"]) == len(paused["tool_results"])
    assert {event["status"] for event in paused["audit_events"]} == {"complete"}

    completed = resume_controlled_investigation(
        graph,
        HumanReviewDecision(action="approve", reviewer_reference="reviewer-1"),
        thread_id="case-1",
    )

    assert completed["review_decision"]["action"] == "approve"
    assert completed["final_status"] == "approved_for_downstream_human_process"


def test_tool_and_review_inputs_reject_unsafe_shapes() -> None:
    with pytest.raises(ValidationError):
        BoundedSubgraphInput(alert_id="alert", hops=3, max_edges=20)
    with pytest.raises(ValidationError):
        HumanReviewDecision(action="edit", reviewer_reference="reviewer", note=None)


def test_tool_failure_returns_minimized_audit_event() -> None:
    evidence = build_mock_evidence_package()
    registry = InvestigationToolRegistry(_retriever())

    result, event = registry.execute(
        InvestigationToolCall(
            name="get_feature_snapshot",
            arguments={"alert_id": "different-alert"},
        ),
        evidence,
    )

    assert result.output == {"error_code": "ValueError"}
    assert event.status == "failed"
    assert event.error_code == "ValueError"
    assert event.output_keys == ["error_code"]


def test_sqlite_checkpoint_resumes_after_reopen(tmp_path: Path) -> None:
    evidence = build_mock_evidence_package()
    retriever = _retriever()
    database = tmp_path / "agent.sqlite"
    first_saver = create_sqlite_checkpointer(database)
    first_graph = build_controlled_investigation_graph(
        InvestigationToolRegistry(retriever),
        retriever=retriever,
        checkpointer=first_saver,
    )
    paused = start_controlled_investigation(first_graph, evidence, thread_id="durable-case")
    assert paused["__interrupt__"]
    first_saver.conn.close()

    second_saver = create_sqlite_checkpointer(database)
    try:
        second_graph = build_controlled_investigation_graph(
            InvestigationToolRegistry(retriever),
            retriever=retriever,
            checkpointer=second_saver,
        )
        completed = resume_controlled_investigation(
            second_graph,
            HumanReviewDecision(action="reject", reviewer_reference="reviewer-2"),
            thread_id="durable-case",
        )
        assert completed["final_status"] == "rejected_by_human_reviewer"
    finally:
        second_saver.conn.close()
