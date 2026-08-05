"""Single-agent LangGraph workflow that drafts only from supplied evidence."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from aml_evidence_graph.evidence.package import (
    AnnotationUsage,
    FactValidationResult,
    InvestigationAnnotation,
    InvestigationReport,
    RiskEvidencePackage,
    SarDraft,
    TypologyReference,
    add_usage,
)
from aml_evidence_graph.evidence.typology import LocalBM25TypologyRetriever
from aml_evidence_graph.investigation.llm import (
    AnnotationProviderError,
    EvidenceAnnotationClient,
    validate_annotation,
)

MAX_TOOL_CALLS = 4


class InvestigationState(TypedDict, total=False):
    """State passed through the bounded single-agent investigation workflow."""

    evidence: RiskEvidencePackage
    retrieved_typologies: list[TypologyReference]
    trace_id: str
    node_timeline: Annotated[list[dict[str, object]], add]
    annotation: InvestigationAnnotation | None
    annotation_error: str | None
    unusable_call_usage: AnnotationUsage | None
    external_call_attempts: int
    external_usage_complete: bool
    retrieval_error: str | None
    fact_validation: FactValidationResult | None
    tool_call_count: int
    report: InvestigationReport


# Every observed state change this chain can report, so a trace consumer can rely on a
# closed vocabulary instead of parsing free text.
_NODE_STATE_CHANGES: dict[str, str] = {
    "retrieve_typologies": "typologies_retrieved",
    "fact_check": "references_deduplicated",
    "annotate": "annotation_attempted",
    "validate_annotation": "annotation_validated",
    "draft_report": "draft_rendered",
}


def _timeline_event(
    node: str,
    started: float,
    *,
    state_change: str,
    status: Literal["complete", "failed"] = "complete",
) -> dict[str, object]:
    """Build one span in the same shape the controlled workflow already emits."""
    return {
        "node": node,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "duration_ms": (time.perf_counter() - started) * 1_000,
        "status": status,
        "state_change": state_change,
    }


def _traced_node(
    node: str,
    function: Callable[[InvestigationState], InvestigationState],
) -> Callable[[InvestigationState], InvestigationState]:
    """Time one node and append its span without changing the node's own contract."""

    def wrapped(state: InvestigationState) -> InvestigationState:
        started = time.perf_counter()
        try:
            update = dict(function(state))
        except Exception:
            # The span is recorded before propagating so a failed run still has a trace.
            state.setdefault("node_timeline", []).append(
                _timeline_event(node, started, state_change="node_failed", status="failed")
            )
            raise
        update["node_timeline"] = [
            _timeline_event(
                node,
                started,
                state_change=_NODE_STATE_CHANGES.get(node, "state_updated"),
            )
        ]
        return update

    return wrapped


def _query_from_evidence(evidence: RiskEvidencePackage) -> str:
    values = [
        *[rule.rule_id for rule in evidence.rule_hits],
        *[feature.name for feature in evidence.key_features],
        *evidence.uncertainty_notes,
    ]
    return " ".join(values) or "transaction risk investigation"


def _retrieve_typologies(
    state: InvestigationState,
    *,
    retriever: LocalBM25TypologyRetriever,
) -> InvestigationState:
    if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
        return {"retrieved_typologies": []}
    evidence = state["evidence"]
    try:
        documents = retriever.retrieve(_query_from_evidence(evidence))
    except Exception:
        return {
            "retrieved_typologies": [],
            "retrieval_error": "typology_retrieval_unavailable",
            "tool_call_count": state.get("tool_call_count", 0) + 1,
        }
    references = [
        TypologyReference(
            typology_id=document.typology_id,
            version=document.version,
            title=document.title,
            source=document.source,
        )
        for document in documents
    ]
    return {
        "retrieved_typologies": references,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
    }


def _fact_check(state: InvestigationState) -> InvestigationState:
    """Deduplicate retrieved references without mutating the original evidence package."""
    evidence = state["evidence"]
    references = state.get("retrieved_typologies", [])
    if evidence.typology_references:
        known_ids = {reference.typology_id for reference in evidence.typology_references}
        references = [
            *evidence.typology_references,
            *[reference for reference in references if reference.typology_id not in known_ids],
        ]
    return {"retrieved_typologies": references}


def _annotate(
    state: InvestigationState,
    *,
    annotator: EvidenceAnnotationClient | None,
) -> InvestigationState:
    """Optionally ask an external model for non-factual analysis after retrieval."""
    if annotator is None:
        return {"annotation": None, "external_call_attempts": 0}

    def usage_complete(attempts: int, *usages: AnnotationUsage | None) -> bool:
        """True when every attempt reported usage, so a summed total is a real bill."""
        return attempts == sum(usage is not None for usage in usages)

    try:
        annotation = annotator.annotate(
            state["evidence"],
            state.get("retrieved_typologies", []),
        )
    except AnnotationProviderError as error:
        return {
            "annotation": None,
            "annotation_error": error.category,
            # Billed but unusable: keep the tokens so cost accounting stays complete.
            # A failed retry bills twice, so both attempts are totalled here.
            "unusable_call_usage": add_usage(error.superseded_usage, error.usage),
            "external_call_attempts": error.attempts,
            "external_usage_complete": usage_complete(
                error.attempts,
                error.usage,
                *((error.superseded_usage,) if error.attempts > 1 else ()),
            ),
        }
    except Exception:
        return {
            "annotation": None,
            "annotation_error": "external_annotation_error",
            "unusable_call_usage": None,
            # A non-provider exception gives no attempt count; one call was still tried.
            "external_call_attempts": 1,
            "external_usage_complete": False,
        }
    return {
        "annotation": annotation,
        "annotation_error": None,
        # Present only when a truncated attempt was discarded before this one succeeded.
        "unusable_call_usage": annotation.superseded_usage,
        "external_call_attempts": annotation.attempt_count,
        "external_usage_complete": usage_complete(
            annotation.attempt_count,
            annotation.usage,
            *((annotation.superseded_usage,) if annotation.attempt_count > 1 else ()),
        ),
    }


def _validate_annotation(state: InvestigationState) -> InvestigationState:
    """Validate all LLM field references before a draft is exposed to a reviewer."""
    annotation = state.get("annotation")
    if annotation is None:
        return {"fact_validation": FactValidationResult(valid=True)}
    result = validate_annotation(
        annotation,
        evidence=state["evidence"],
        references=state.get("retrieved_typologies", []),
    )
    return {"fact_validation": result}


def _draft_report(state: InvestigationState) -> InvestigationState:
    """Render a deterministic, review-only report without generating unsupported facts."""
    evidence = state["evidence"]
    references = state.get("retrieved_typologies", evidence.typology_references)
    annotation = state.get("annotation")
    validation = state.get("fact_validation") or FactValidationResult(valid=True)
    uncertainty_notes = list(evidence.uncertainty_notes)
    if state.get("annotation_error") is not None:
        uncertainty_notes.append(
            "External LLM annotation was unavailable "
            f"({state['annotation_error']}); deterministic evidence template used."
        )
    if state.get("retrieval_error") == "typology_retrieval_unavailable":
        uncertainty_notes.append(
            "Typology retrieval was unavailable; no retrieved typology lead was used."
        )
    score_items = [
        f"{name} probability: {value:.6f}"
        for name, value in sorted(evidence.model_probabilities.items())
    ]
    if evidence.fusion_probability is not None:
        score_items.append(f"fusion probability: {evidence.fusion_probability:.6f}")
    rule_items = [
        f"{rule.rule_id} ({rule.rule_version}) observed {rule.feature}={rule.observed_value:g}; "
        f"threshold {rule.operator} {rule.threshold:g}."
        for rule in evidence.rule_hits
    ]
    feature_items = [
        f"{feature.name}={feature.value} ({feature.source})"
        for feature in evidence.key_features
    ]
    graph_items = (
        [
            "Historical graph evidence: "
            f"source out-degree={evidence.graph_evidence.historical_source_out_degree}, "
            f"destination in-degree={evidence.graph_evidence.historical_destination_in_degree}, "
            f"prior directed edges={evidence.graph_evidence.prior_directed_edge_count}, "
            f"prior reverse edges={evidence.graph_evidence.prior_reverse_edge_count}, "
            f"bounded two-hop paths={evidence.graph_evidence.two_hop_intermediary_count}."
        ]
        if evidence.graph_evidence is not None
        else []
    )
    factual_summary = [*score_items, *rule_items, *feature_items, *graph_items]
    typology_considerations = [
        f"Consider {reference.typology_id} v{reference.version}: {reference.title}. "
        "This retrieval is a lead, not a case classification."
        for reference in references
    ]
    annotation_items = (
        [
            *annotation.analytical_considerations,
            *annotation.recommended_questions,
        ]
        if annotation is not None and validation.valid
        else []
    )
    sar_draft = None
    if validation.valid:
        supporting_refs = [
            *[f"model_probabilities.{name}" for name in sorted(evidence.model_probabilities)],
            *[f"rule_hits[{index}]" for index in range(len(evidence.rule_hits))],
            *[f"key_features[{index}]" for index in range(len(evidence.key_features))],
            *(["graph_evidence"] if evidence.graph_evidence is not None else []),
            *[
                f"typology_references.{reference.typology_id}"
                for reference in references
            ],
        ]
        fund_path_notes = []
        if evidence.graph_evidence is not None:
            fund_path_notes.append(
                "Historical graph evidence summarizes prior directed/reverse edges and "
                "bounded two-hop intermediaries only; it is not a confirmed laundering path."
            )
        sar_draft = SarDraft(
            background=[
                f"Alert {evidence.alert_id} drafted from RiskEvidencePackage schema "
                f"{evidence.schema_version}.",
                "All numeric values below are copied from the evidence package.",
            ],
            observed_behaviors=factual_summary,
            typology_leads=typology_considerations,
            fund_path_notes=fund_path_notes,
            supporting_evidence_refs=supporting_refs,
            pending_verification=[
                *evidence.missing_evidence,
                *evidence.uncertainty_notes,
                "Confirm whether activity warrants a formal SAR filing.",
            ],
        )
    report = InvestigationReport(
        alert_id=evidence.alert_id,
        status=(
            "draft_requires_human_review"
            if validation.valid
            else "rejected_facts"
        ),
        factual_summary=factual_summary,
        typology_considerations=[*typology_considerations, *annotation_items],
        missing_evidence=evidence.missing_evidence,
        uncertainty_notes=uncertainty_notes,
        fact_snapshot=evidence.model_dump(mode="json"),
        review_instruction=(
            "A human investigator must verify the cited evidence and approve, "
            "amend, or reject this draft before any case action."
        ),
        sar_draft=sar_draft,
        llm_annotation=annotation if validation.valid else None,
        fact_validation=validation,
        annotation_error_category=state.get("annotation_error"),
        # A fact-gate rejection drops the annotation from the report, but the call was
        # still billed, so its usage moves to the unusable-call basis instead of vanishing.
        # Any attempt discarded before it is already on that basis and stays there.
        unusable_call_usage=(
            state.get("unusable_call_usage")
            if validation.valid
            else add_usage(
                state.get("unusable_call_usage"),
                annotation.usage if annotation is not None else None,
            )
        ),
        external_call_attempts=state.get("external_call_attempts", 0),
        external_usage_complete=state.get("external_usage_complete", True),
        tool_call_count=state.get("tool_call_count", 0),
    )
    return {"report": report}


def build_investigation_graph(
    retriever: LocalBM25TypologyRetriever,
    *,
    annotator: EvidenceAnnotationClient | None = None,
) -> StateGraph[InvestigationState]:
    """Build a single deterministic Agent workflow with explicit fact-check stage."""
    workflow = StateGraph(InvestigationState)
    workflow.add_node(
        "retrieve_typologies",
        _traced_node(
            "retrieve_typologies",
            lambda state: _retrieve_typologies(state, retriever=retriever),
        ),
    )
    workflow.add_node("fact_check", _traced_node("fact_check", _fact_check))
    workflow.add_node(
        "annotate",
        _traced_node("annotate", lambda state: _annotate(state, annotator=annotator)),
    )
    workflow.add_node(
        "validate_annotation",
        _traced_node("validate_annotation", _validate_annotation),
    )
    workflow.add_node("draft_report", _traced_node("draft_report", _draft_report))
    workflow.add_edge(START, "retrieve_typologies")
    workflow.add_edge("retrieve_typologies", "fact_check")
    workflow.add_edge("fact_check", "annotate")
    workflow.add_edge("annotate", "validate_annotation")
    workflow.add_edge("validate_annotation", "draft_report")
    workflow.add_edge("draft_report", END)
    return workflow


class InvestigationTrace(TypedDict):
    """One correlated run record: identity plus the span list, no evidence bodies."""

    trace_id: str
    alert_id: str
    node_timeline: list[dict[str, object]]
    total_duration_ms: float
    annotation_error_category: str | None
    report_status: str


def run_investigation_traced(
    evidence: RiskEvidencePackage,
    *,
    retriever: LocalBM25TypologyRetriever,
    annotator: EvidenceAnnotationClient | None = None,
    trace_id: str | None = None,
) -> tuple[InvestigationReport, InvestigationTrace]:
    """Run the workflow and return the draft plus its correlated node trace.

    Kept separate from :func:`run_investigation` so the API response model does not
    grow operational spans that reviewers never need.
    """
    graph = build_investigation_graph(retriever, annotator=annotator).compile()
    resolved_trace_id = trace_id or f"trace-{uuid.uuid4().hex}"
    state = graph.invoke({"evidence": evidence, "trace_id": resolved_trace_id})
    report: InvestigationReport = state["report"]
    timeline = list(state.get("node_timeline", []))
    trace: InvestigationTrace = {
        "trace_id": resolved_trace_id,
        "alert_id": report.alert_id,
        "node_timeline": timeline,
        "total_duration_ms": sum(float(span["duration_ms"]) for span in timeline),
        "annotation_error_category": report.annotation_error_category,
        "report_status": report.status,
    }
    return report, trace


def run_investigation(
    evidence: RiskEvidencePackage,
    *,
    retriever: LocalBM25TypologyRetriever,
    annotator: EvidenceAnnotationClient | None = None,
) -> InvestigationReport:
    """Run the evidence-bound workflow and return a human-review draft."""
    report, _ = run_investigation_traced(
        evidence,
        retriever=retriever,
        annotator=annotator,
    )
    return report
