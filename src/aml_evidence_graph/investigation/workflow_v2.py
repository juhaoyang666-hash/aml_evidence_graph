"""Controlled LangGraph workflow with structured tools and resumable human review."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, model_validator

from aml_evidence_graph.evidence.package import RiskEvidencePackage
from aml_evidence_graph.investigation.llm import EvidenceAnnotationClient
from aml_evidence_graph.investigation.tools import (
    InvestigationToolCall,
    InvestigationToolRegistry,
    plan_read_only_tool_calls,
)
from aml_evidence_graph.investigation.workflow import run_investigation


class HumanReviewDecision(BaseModel):
    """Explicit reviewer action required before a controlled run can complete."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "edit", "reject"]
    reviewer_reference: str
    note: str | None = None

    @model_validator(mode="after")
    def require_edit_note(self) -> HumanReviewDecision:
        if self.action == "edit" and not self.note:
            raise ValueError("An edit decision requires a note.")
        return self


class ControlledInvestigationState(TypedDict, total=False):
    """JSON-serializable state suitable for memory or SQLite checkpoints."""

    evidence: dict[str, object]
    tool_calls: list[dict[str, object]]
    tool_results: list[dict[str, object]]
    audit_events: list[dict[str, object]]
    report: dict[str, object]
    review_decision: dict[str, object]
    final_status: str


def build_controlled_investigation_graph(
    registry: InvestigationToolRegistry,
    *,
    retriever: Any,
    annotator: EvidenceAnnotationClient | None = None,
    checkpointer: Any = None,
) -> Any:
    """Compile a bounded workflow; callers choose memory or durable SQLite persistence."""

    def route_tools(state: ControlledInvestigationState) -> ControlledInvestigationState:
        evidence = RiskEvidencePackage.model_validate(state["evidence"])
        calls = plan_read_only_tool_calls(evidence)
        return {"tool_calls": [call.model_dump(mode="json") for call in calls]}

    def execute_tools(state: ControlledInvestigationState) -> ControlledInvestigationState:
        evidence = RiskEvidencePackage.model_validate(state["evidence"])
        results: list[dict[str, object]] = []
        events: list[dict[str, object]] = []
        for raw_call in state.get("tool_calls", [])[:4]:
            call = InvestigationToolCall.model_validate(raw_call)
            result, event = registry.execute(call, evidence)
            results.append(result.model_dump(mode="json"))
            events.append(event.model_dump(mode="json"))
        return {"tool_results": results, "audit_events": events}

    def draft_report(state: ControlledInvestigationState) -> ControlledInvestigationState:
        evidence = RiskEvidencePackage.model_validate(state["evidence"])
        report = run_investigation(evidence, retriever=retriever, annotator=annotator)
        report = report.model_copy(update={"tool_call_count": len(state.get("tool_results", []))})
        return {"report": report.model_dump(mode="json")}

    def request_review(state: ControlledInvestigationState) -> ControlledInvestigationState:
        report = state["report"]
        resumed = interrupt(
            {
                "kind": "human_review",
                "alert_id": report["alert_id"],
                "report_status": report["status"],
                "allowed_decisions": ["approve", "edit", "reject"],
                "message": "Review the evidence-bound draft before any case action.",
            }
        )
        decision = HumanReviewDecision.model_validate(resumed)
        return {"review_decision": decision.model_dump(mode="json")}

    def finalize(state: ControlledInvestigationState) -> ControlledInvestigationState:
        decision = HumanReviewDecision.model_validate(state["review_decision"])
        status = {
            "approve": "approved_for_downstream_human_process",
            "edit": "returned_for_human_edit",
            "reject": "rejected_by_human_reviewer",
        }[decision.action]
        return {"final_status": status}

    workflow = StateGraph(ControlledInvestigationState)
    workflow.add_node("route_tools", route_tools)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("draft_report", draft_report)
    workflow.add_node("human_review", request_review)
    workflow.add_node("finalize", finalize)
    workflow.add_edge(START, "route_tools")
    workflow.add_edge("route_tools", "execute_tools")
    workflow.add_edge("execute_tools", "draft_report")
    workflow.add_edge("draft_report", "human_review")
    workflow.add_edge("human_review", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer)


def start_controlled_investigation(
    graph: Any,
    evidence: RiskEvidencePackage,
    *,
    thread_id: str,
) -> dict[str, object]:
    """Run through draft creation and pause at the mandatory reviewer interrupt."""
    if not thread_id.strip():
        raise ValueError("thread_id must be non-empty.")
    return graph.invoke(
        {"evidence": evidence.model_dump(mode="json")},
        config={"configurable": {"thread_id": thread_id}},
    )


def resume_controlled_investigation(
    graph: Any,
    decision: HumanReviewDecision,
    *,
    thread_id: str,
) -> dict[str, object]:
    """Resume the exact checkpoint with an approve/edit/reject decision."""
    return graph.invoke(
        Command(resume=decision.model_dump(mode="json")),
        config={"configurable": {"thread_id": thread_id}},
    )


def create_sqlite_checkpointer(path: Path) -> Any:
    """Create a local durable saver; requires patched langgraph-checkpoint-sqlite."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as error:
        raise RuntimeError(
            "Install the 'agent' extra (langgraph-checkpoint-sqlite>=3.0.1)."
        ) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection)
