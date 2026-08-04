"""The Golden investigation chain must emit spans in the controlled chain's shape."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from aml_evidence_graph.evidence.package import RiskEvidencePackage
from aml_evidence_graph.evidence.typology import LocalBM25TypologyRetriever, TypologyDocument
from aml_evidence_graph.investigation.golden import GoldenCase, evaluate_golden_set
from aml_evidence_graph.investigation.llm import ECNUAnnotationClient
from aml_evidence_graph.investigation.workflow import (
    run_investigation,
    run_investigation_traced,
)

EXPECTED_NODES = [
    "retrieve_typologies",
    "fact_check",
    "annotate",
    "validate_annotation",
    "draft_report",
]
# Same keys the controlled workflow already emits, so one consumer reads both chains.
SPAN_KEYS = {"node", "timestamp_utc", "duration_ms", "status", "state_change"}


def _evidence(alert_id: str = "alert-trace") -> RiskEvidencePackage:
    return RiskEvidencePackage(
        alert_id=alert_id,
        generated_at=datetime(2026, 8, 4, tzinfo=UTC),
        transaction_id="txn-trace",
        event_timestamp=datetime(2023, 7, 1, tzinfo=UTC),
        model_probabilities={"catboost": 0.8},
    )


def _retriever() -> LocalBM25TypologyRetriever:
    return LocalBM25TypologyRetriever(
        [
            TypologyDocument(
                typology_id="T-TRACE",
                version="1",
                title="Trace typology",
                source="test",
                body="Transaction risk investigation.",
            )
        ]
    )


def test_trace_records_every_node_in_execution_order() -> None:
    report, trace = run_investigation_traced(
        _evidence(), retriever=_retriever(), trace_id="trace-fixed"
    )

    assert trace["trace_id"] == "trace-fixed"
    assert trace["alert_id"] == report.alert_id
    assert [span["node"] for span in trace["node_timeline"]] == EXPECTED_NODES
    assert trace["report_status"] == report.status
    assert trace["annotation_error_category"] is report.annotation_error_category


def test_every_span_carries_the_controlled_chain_shape() -> None:
    _, trace = run_investigation_traced(_evidence(), retriever=_retriever())

    for span in trace["node_timeline"]:
        assert set(span) == SPAN_KEYS
        assert span["status"] == "complete"
        assert isinstance(span["duration_ms"], float)
        assert span["duration_ms"] >= 0
        # A closed vocabulary means consumers never parse free text.
        assert span["state_change"] != ""
        datetime.fromisoformat(str(span["timestamp_utc"]))


def test_total_duration_is_the_sum_of_spans() -> None:
    _, trace = run_investigation_traced(_evidence(), retriever=_retriever())

    expected = sum(float(span["duration_ms"]) for span in trace["node_timeline"])
    assert trace["total_duration_ms"] == pytest.approx(expected)


def test_generated_trace_ids_are_unique_per_run() -> None:
    _, first = run_investigation_traced(_evidence(), retriever=_retriever())
    _, second = run_investigation_traced(_evidence(), retriever=_retriever())

    assert first["trace_id"] != second["trace_id"]


def test_untraced_entrypoint_keeps_its_report_only_contract() -> None:
    """The API depends on this signature; spans must not leak into the response model."""
    report = run_investigation(_evidence(), retriever=_retriever())

    assert report.alert_id == "alert-trace"
    assert "node_timeline" not in report.model_dump(mode="json")


def test_failed_external_call_still_produces_a_complete_trace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    annotator = ECNUAnnotationClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    _, trace = run_investigation_traced(
        _evidence(), retriever=_retriever(), annotator=annotator
    )

    # A provider failure is handled inside the node, so the chain still completes.
    assert [span["node"] for span in trace["node_timeline"]] == EXPECTED_NODES
    assert trace["annotation_error_category"] == "annotation_json_invalid"


def test_golden_run_writes_one_trace_per_case(tmp_path: Path) -> None:
    del tmp_path
    cases = [
        GoldenCase(case_id=f"trace-case-{index}", evidence=_evidence(f"alert-{index}"))
        for index in range(3)
    ]
    traces: list[dict[str, object]] = []

    summary = evaluate_golden_set(cases, retriever=_retriever(), trace_sink=traces)

    assert summary.case_count == 3
    assert [trace["case_id"] for trace in traces] == [case.case_id for case in cases]
    assert [trace["trace_id"] for trace in traces] == [
        f"golden-{case.case_id}" for case in cases
    ]
    # Each ledger line must be serializable on its own for JSONL output.
    for trace in traces:
        assert json.loads(json.dumps(trace, ensure_ascii=False))["case_id"]


def test_golden_run_without_a_sink_still_succeeds() -> None:
    summary = evaluate_golden_set(
        [GoldenCase(case_id="no-sink", evidence=_evidence())],
        retriever=_retriever(),
    )

    assert summary.case_count == 1
