"""Read-only, schema-bound tools for controlled investigation workflows."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from aml_evidence_graph.evidence.package import RiskEvidencePackage
from aml_evidence_graph.evidence.typology import TypologyDocument


class TypologyRetriever(Protocol):
    """Compatibility protocol implemented by BM25 and hybrid retrievers."""

    def retrieve(self, query: str, *, limit: int = 3) -> list[TypologyDocument]: ...


class FeatureSnapshotInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_id: str


class BoundedSubgraphInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_id: str
    hops: int = Field(default=2, ge=1, le=2)
    max_edges: int = Field(default=20, ge=1, le=50)


class TypologySearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_id: str
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


class InvestigationToolCall(BaseModel):
    """A structured function call; arbitrary code, paths, SQL, and URLs are not accepted."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["get_feature_snapshot", "get_bounded_subgraph", "search_typologies"]
    arguments: dict[str, object]


class InvestigationToolResult(BaseModel):
    """Internal tool output copied only from an existing evidence package or local corpus."""

    model_config = ConfigDict(extra="forbid")

    name: str
    output: dict[str, object]


class ToolAuditEvent(BaseModel):
    """Audit metadata deliberately excludes full feature values and document bodies."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    timestamp_utc: datetime
    alert_id: str
    tool_name: str
    argument_names: list[str]
    output_keys: list[str]
    duration_ms: float = Field(ge=0)
    status: Literal["complete", "failed"]
    error_code: str | None = None


class InvestigationToolRegistry:
    """Execute the three approved read-only tools against one evidence package."""

    def __init__(self, retriever: TypologyRetriever) -> None:
        self.retriever = retriever

    @staticmethod
    def _assert_alert(expected: str, supplied: str) -> None:
        if supplied != expected:
            raise ValueError("Tool alert_id does not match the active evidence package.")

    def execute(
        self,
        call: InvestigationToolCall,
        evidence: RiskEvidencePackage,
    ) -> tuple[InvestigationToolResult, ToolAuditEvent]:
        """Validate arguments, execute locally, and return a value-minimized audit event."""
        started = time.perf_counter()
        status: Literal["complete", "failed"] = "complete"
        error_code = None
        output: dict[str, object] = {}
        try:
            if call.name == "get_feature_snapshot":
                arguments = FeatureSnapshotInput.model_validate(call.arguments)
                self._assert_alert(evidence.alert_id, arguments.alert_id)
                output = {
                    "features": [item.model_dump(mode="json") for item in evidence.key_features],
                    "source_versions": evidence.source_versions,
                    "event_timestamp": evidence.event_timestamp.isoformat(),
                }
            elif call.name == "get_bounded_subgraph":
                arguments = BoundedSubgraphInput.model_validate(call.arguments)
                self._assert_alert(evidence.alert_id, arguments.alert_id)
                graph = evidence.graph_evidence
                output = {
                    "hops": arguments.hops,
                    "max_edges": arguments.max_edges,
                    "graph_evidence": graph.model_dump(mode="json") if graph else None,
                    "missing": graph is None,
                }
            else:
                arguments = TypologySearchInput.model_validate(call.arguments)
                self._assert_alert(evidence.alert_id, arguments.alert_id)
                documents = self.retriever.retrieve(arguments.query, limit=arguments.top_k)
                output = {
                    "references": [
                        {
                            "typology_id": document.typology_id,
                            "version": document.version,
                            "title": document.title,
                            "source": document.source,
                        }
                        for document in documents
                    ]
                }
        except Exception as error:
            status = "failed"
            error_code = type(error).__name__
            output = {"error_code": error_code}
        finally:
            event = ToolAuditEvent(
                event_id=f"tool-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}",
                timestamp_utc=datetime.now(UTC),
                alert_id=evidence.alert_id,
                tool_name=call.name,
                argument_names=sorted(call.arguments),
                output_keys=sorted(output),
                duration_ms=(time.perf_counter() - started) * 1_000,
                status=status,
                error_code=error_code,
            )
        return InvestigationToolResult(name=call.name, output=output), event


def plan_read_only_tool_calls(evidence: RiskEvidencePackage) -> list[InvestigationToolCall]:
    """Deterministic baseline router used before introducing any model-based routing."""
    calls: list[InvestigationToolCall] = []
    if evidence.key_features:
        calls.append(
            InvestigationToolCall(
                name="get_feature_snapshot",
                arguments={"alert_id": evidence.alert_id},
            )
        )
    if evidence.graph_evidence is not None:
        calls.append(
            InvestigationToolCall(
                name="get_bounded_subgraph",
                arguments={"alert_id": evidence.alert_id, "hops": 2, "max_edges": 20},
            )
        )
    query_parts = [
        *[rule.rule_id for rule in evidence.rule_hits],
        *[feature.name for feature in evidence.key_features],
        *evidence.uncertainty_notes,
    ]
    calls.append(
        InvestigationToolCall(
            name="search_typologies",
            arguments={
                "alert_id": evidence.alert_id,
                "query": " ".join(query_parts) or "transaction risk investigation",
                "top_k": 3,
            },
        )
    )
    return calls[:4]
