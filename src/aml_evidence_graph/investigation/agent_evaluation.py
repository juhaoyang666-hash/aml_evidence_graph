"""Evaluate the controlled investigation agent against versioned synthetic cases."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aml_evidence_graph.demo.mock_data import build_mock_evidence_package
from aml_evidence_graph.evidence.package import GraphEvidence, RiskEvidencePackage
from aml_evidence_graph.evidence.typology import TypologyDocument
from aml_evidence_graph.investigation.tools import (
    BoundedSubgraphInput,
    FeatureSnapshotInput,
    InvestigationToolCall,
    InvestigationToolRegistry,
    TypologySearchInput,
)
from aml_evidence_graph.investigation.workflow_v2 import (
    HumanReviewDecision,
    build_controlled_investigation_graph,
    create_sqlite_checkpointer,
    resume_controlled_investigation,
    start_controlled_investigation,
)


class AgentGoldenCase(BaseModel):
    """One auditable, synthetic routing, tool, review, or recovery expectation."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    category: Literal["routing", "tool", "review", "recovery"]
    variant: str = "base"
    expected_tools: list[str] = Field(default_factory=list)
    raw_call: dict[str, object] | None = None
    expected_outcome: str
    action: str | None = None
    note: str | None = None
    failure_mode: Literal["none", "once", "always"] = "none"
    checkpointer: Literal["memory", "sqlite"] = "memory"
    tags: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class AgentCaseResult:
    case_id: str
    category: str
    passed: bool
    expected_outcome: str
    observed_outcome: str
    tool_selection_correct: bool | None
    parameter_validity_correct: bool | None
    fact_consistent: bool | None
    recovery_success: bool | None
    latency_ms: float
    node_timeline: list[dict[str, object]]
    tool_audit: list[dict[str, object]]
    sources: list[str]
    review_decision: dict[str, object] | None
    tags: list[str]


class _SyntheticRetriever:
    def __init__(self, failure_mode: str = "none") -> None:
        self.failure_mode = failure_mode
        self.calls = 0

    def retrieve(self, query: str, *, limit: int = 3) -> list[TypologyDocument]:
        self.calls += 1
        if self.failure_mode == "always" or (
            self.failure_mode == "once" and self.calls == 1
        ):
            raise TimeoutError("synthetic retrieval failure")
        return [
            TypologyDocument(
                typology_id="TYPOLOGY-STRUCTURING",
                version="2026.1",
                title="Structuring / Smurfing",
                body="Repeated split transfers are an investigation lead.",
                source="agent-golden-synthetic",
            )
        ][:limit]


def _evidence_variant(name: str) -> RiskEvidencePackage:
    evidence = build_mock_evidence_package()
    graph = GraphEvidence(
        source_node_index=101,
        destination_node_index=202,
        historical_source_out_degree=3,
        historical_destination_in_degree=4,
        prior_directed_edge_count=2,
        prior_reverse_edge_count=1,
        two_hop_intermediary_count=1,
        two_hop_intermediary_node_indices=[303],
        snapshot_as_of=evidence.event_timestamp,
        interpretation_limit="Synthetic history-only graph evidence; not a confirmed path.",
    )
    updates: dict[str, object] = {}
    if name in {"minimal", "graph_only"}:
        updates["key_features"] = []
    if name in {"full", "graph_only", "full_no_refs", "full_no_uncertainty"}:
        updates["graph_evidence"] = graph
    if name in {"no_refs", "full_no_refs"}:
        updates["typology_references"] = []
    if name in {"no_uncertainty", "full_no_uncertainty"}:
        updates["uncertainty_notes"] = []
    if name == "single_feature":
        updates["key_features"] = evidence.key_features[:1]
    if name == "no_missing":
        updates["missing_evidence"] = []
    if name not in {
        "base",
        "minimal",
        "graph_only",
        "full",
        "no_refs",
        "full_no_refs",
        "no_uncertainty",
        "full_no_uncertainty",
        "single_feature",
        "no_missing",
    }:
        raise ValueError(f"Unknown evidence variant: {name}")
    return evidence.model_copy(update=updates, deep=True)


def _validate_call_arguments(call: InvestigationToolCall) -> None:
    models = {
        "get_feature_snapshot": FeatureSnapshotInput,
        "get_bounded_subgraph": BoundedSubgraphInput,
        "search_typologies": TypologySearchInput,
    }
    models[call.name].model_validate(call.arguments)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _run_workflow_case(case: AgentGoldenCase) -> AgentCaseResult:
    started = time.perf_counter()
    evidence = _evidence_variant(case.variant)
    retriever = _SyntheticRetriever(case.failure_mode)
    saver: Any
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if case.checkpointer == "sqlite":
        temporary = tempfile.TemporaryDirectory()
        sqlite_path = Path(temporary.name) / "agent.sqlite"
        saver = create_sqlite_checkpointer(sqlite_path)
    else:
        saver = InMemorySaver()
    graph = build_controlled_investigation_graph(
        InvestigationToolRegistry(retriever), retriever=retriever, checkpointer=saver
    )
    thread_id = f"golden-{case.case_id}"
    try:
        paused = start_controlled_investigation(graph, evidence, thread_id=thread_id)
        planned = [item["name"] for item in paused.get("tool_calls", [])]
        tool_correct = planned == case.expected_tools if case.expected_tools else None
        fact_consistent = paused["report"]["fact_snapshot"] == evidence.model_dump(mode="json")
        if case.category == "routing":
            observed = "routed"
            passed = tool_correct is True and fact_consistent
            recovery = None
            state = paused
        else:
            if case.checkpointer == "sqlite":
                saver.conn.close()
                saver = create_sqlite_checkpointer(sqlite_path)
                graph = build_controlled_investigation_graph(
                    InvestigationToolRegistry(retriever),
                    retriever=retriever,
                    checkpointer=saver,
                )
            try:
                decision = HumanReviewDecision.model_validate(
                    {
                        "action": case.action,
                        "reviewer_reference": "agent-golden-reviewer",
                        "note": case.note,
                    }
                )
            except ValidationError:
                observed = "review_validation_failed"
                passed = case.expected_outcome == observed
                recovery = None
                state = paused
            else:
                state = resume_controlled_investigation(graph, decision, thread_id=thread_id)
                observed = str(state["final_status"])
                if case.category == "recovery":
                    failure_audited = any(
                        event["status"] == "failed"
                        for event in state.get("audit_events", [])
                    )
                    degradation_declared = any(
                        "Typology retrieval was unavailable" in note
                        for note in state["report"]["uncertainty_notes"]
                    )
                    recovery = (
                        observed == case.expected_outcome
                        and failure_audited
                        and (case.failure_mode != "always" or degradation_declared)
                    )
                else:
                    recovery = None
                passed = observed == case.expected_outcome and fact_consistent
                if recovery is not None:
                    passed = passed and recovery
        return AgentCaseResult(
            case_id=case.case_id,
            category=case.category,
            passed=passed,
            expected_outcome=case.expected_outcome,
            observed_outcome=observed,
            tool_selection_correct=tool_correct,
            parameter_validity_correct=None,
            fact_consistent=fact_consistent,
            recovery_success=recovery,
            latency_ms=(time.perf_counter() - started) * 1_000,
            node_timeline=list(state.get("node_timeline", [])),
            tool_audit=list(state.get("audit_events", [])),
            sources=[f"{key}:{value}" for key, value in evidence.source_versions.items()],
            review_decision=state.get("review_decision"),
            tags=case.tags,
        )
    finally:
        if case.checkpointer == "sqlite":
            saver.conn.close()
        if temporary is not None:
            temporary.cleanup()


def _run_tool_case(case: AgentGoldenCase) -> AgentCaseResult:
    started = time.perf_counter()
    evidence = _evidence_variant(case.variant)
    observed = "call_validation_failed"
    audit: list[dict[str, object]] = []
    valid = False
    try:
        call = InvestigationToolCall.model_validate(case.raw_call)
        _validate_call_arguments(call)
    except ValidationError:
        pass
    else:
        valid = True
        result, event = InvestigationToolRegistry(_SyntheticRetriever()).execute(call, evidence)
        observed = (
            "tool_complete"
            if event.status == "complete"
            else f"tool_failed:{event.error_code}"
        )
        audit = [event.model_dump(mode="json")]
        if result.output.get("error_code") and event.status == "complete":
            observed = "invalid_audit_state"
    return AgentCaseResult(
        case_id=case.case_id,
        category=case.category,
        passed=observed == case.expected_outcome,
        expected_outcome=case.expected_outcome,
        observed_outcome=observed,
        tool_selection_correct=None,
        parameter_validity_correct=(valid == (case.expected_outcome != "call_validation_failed")),
        fact_consistent=None,
        recovery_success=None,
        latency_ms=(time.perf_counter() - started) * 1_000,
        node_timeline=[],
        tool_audit=audit,
        sources=[f"{key}:{value}" for key, value in evidence.source_versions.items()],
        review_decision=None,
        tags=case.tags,
    )


def evaluate_agent_cases(cases: list[AgentGoldenCase]) -> dict[str, object]:
    """Run cases and return aggregate metrics plus complete per-case audit evidence."""
    results = [
        _run_tool_case(case) if case.category == "tool" else _run_workflow_case(case)
        for case in cases
    ]
    latencies = [item.latency_ms for item in results]

    def ratio(field: str) -> float | None:
        values = [getattr(item, field) for item in results if getattr(item, field) is not None]
        return sum(bool(value) for value in values) / len(values) if values else None

    return {
        "schema_version": "1.0",
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "case_count": len(results),
        "pass_rate": sum(item.passed for item in results) / len(results),
        "tool_selection_accuracy": ratio("tool_selection_correct"),
        "parameter_validity_accuracy": ratio("parameter_validity_correct"),
        "fact_consistency_rate": ratio("fact_consistent"),
        "recovery_success_rate": ratio("recovery_success"),
        "latency_ms": {
            "median": median(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "token_cost_coverage": {
            "cases_with_provider_usage": 0,
            "case_count": len(results),
            "coverage": 0.0,
            "reason": "Deterministic no-LLM baseline; no provider token or cost claim is made.",
        },
        "bad_cases": [asdict(item) for item in results if not item.passed],
        "case_results": [asdict(item) for item in results],
    }


def load_agent_cases(path: Path) -> list[AgentGoldenCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Agent Golden file must contain a JSON list.")
    return [AgentGoldenCase.model_validate(item) for item in raw]
