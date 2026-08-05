"""A truncated annotation may be retried once, and both attempts must stay on the bill.

The retry is deliberately narrow: only a provider-declared `finish_reason="length"`
qualifies. These tests pin that boundary, because widening it would inflate the parse
rate while hiding the provider instability this project measures on purpose.
See golden/llm_retry_policy_v1.json.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import yaml

from aml_evidence_graph.evidence.package import RiskEvidencePackage
from aml_evidence_graph.evidence.typology import LocalBM25TypologyRetriever, TypologyDocument
from aml_evidence_graph.investigation.golden import GoldenCase, evaluate_golden_set
from aml_evidence_graph.investigation.llm import (
    DEFAULT_PROMPT_CONFIGURATION,
    AnnotationProviderError,
    ECNUAnnotationClient,
    load_prompt_configuration,
)
from aml_evidence_graph.investigation.workflow import run_investigation

INPUT_PRICE = 3.0
OUTPUT_PRICE = 12.0
BASE_MAX_TOKENS = 500
RETRY_MAX_TOKENS = 1000

V6_PATH = Path("configs/prompts/ecnu-risk-evidence-v6.yaml")
V7_PATH = Path("configs/prompts/ecnu-risk-evidence-v7.yaml")

VALID_ANNOTATION = {
    "evidence_references": ["model_probabilities.catboost"],
    "analytical_considerations": ["Consider corroborating the observed pattern."],
    "recommended_questions": ["What authorized records could corroborate the pattern?"],
}
# Cut mid-array, exactly how a real length stop leaves the body.
TRUNCATED_BODY = '{"evidence_references": [], "analytical_considerations": ['


def _evidence(alert_id: str = "alert-retry") -> RiskEvidencePackage:
    return RiskEvidencePackage(
        alert_id=alert_id,
        generated_at=datetime(2026, 8, 5, tzinfo=UTC),
        transaction_id="txn-retry",
        event_timestamp=datetime(2023, 7, 1, tzinfo=UTC),
        model_probabilities={"catboost": 0.8},
    )


def _retriever() -> LocalBM25TypologyRetriever:
    return LocalBM25TypologyRetriever(
        [
            TypologyDocument(
                typology_id="T-RETRY",
                version="1",
                title="Retry typology",
                source="test",
                body="Transaction risk investigation.",
            )
        ]
    )


def _response(
    content: str,
    *,
    finish_reason: str | None = None,
    prompt_tokens: int = 700,
    completion_tokens: int = 300,
) -> httpx.Response:
    choice: dict[str, object] = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return httpx.Response(
        200,
        json={
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "choices": [choice],
        },
    )


class _Recorder:
    """Serve a scripted response per attempt and keep every requested token ceiling."""

    def __init__(self, *responses) -> None:
        self._responses = list(responses)
        self.requested_max_tokens: list[int] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        self.requested_max_tokens.append(payload["max_tokens"])
        index = min(len(self.requested_max_tokens) - 1, len(self._responses) - 1)
        outcome = self._responses[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def attempts(self) -> int:
        return len(self.requested_max_tokens)


def _client(handler, *, retry_ceiling: int | None = RETRY_MAX_TOKENS):
    configuration = DEFAULT_PROMPT_CONFIGURATION.__class__(
        version="test-prompt",
        system_instructions="test",
        temperature=0,
        max_tokens=BASE_MAX_TOKENS,
        truncation_retry_max_tokens=retry_ceiling,
    )
    return ECNUAnnotationClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        input_cost_per_million_tokens_usd=INPUT_PRICE,
        output_cost_per_million_tokens_usd=OUTPUT_PRICE,
        prompt_configuration=configuration,
    )


# --- the retry boundary -------------------------------------------------------


def test_truncation_is_retried_once_at_the_wider_ceiling() -> None:
    recorder = _Recorder(
        _response(TRUNCATED_BODY, finish_reason="length"),
        _response(json.dumps(VALID_ANNOTATION)),
    )

    annotation = _client(recorder).annotate(_evidence(), [])

    assert recorder.requested_max_tokens == [BASE_MAX_TOKENS, RETRY_MAX_TOKENS]
    assert annotation.attempt_count == 2
    assert annotation.analytical_considerations == VALID_ANNOTATION[
        "analytical_considerations"
    ]


def test_retry_is_off_unless_a_ceiling_is_configured() -> None:
    """An existing prompt version must keep the exact behavior its holdout measured."""
    recorder = _Recorder(_response(TRUNCATED_BODY, finish_reason="length"))

    with pytest.raises(AnnotationProviderError) as raised:
        _client(recorder, retry_ceiling=None).annotate(_evidence(), [])

    assert raised.value.category == "annotation_truncated"
    assert recorder.attempts == 1


@pytest.mark.parametrize(
    ("outcome", "expected_category"),
    [
        (httpx.TimeoutException("slow"), "timeout"),
        (httpx.ConnectError("boom"), "transport_error"),
        (_response("not json at all"), "annotation_json_invalid"),
        (_response('["not", "an", "object"]'), "annotation_schema_invalid"),
        # A length stop that still decoded is not a truncation failure at all.
        (_response(json.dumps(VALID_ANNOTATION), finish_reason="length"), None),
    ],
)
def test_only_declared_truncation_triggers_a_second_call(
    outcome: object,
    expected_category: str | None,
) -> None:
    recorder = _Recorder(outcome, _response(json.dumps(VALID_ANNOTATION)))
    client = _client(recorder)

    if expected_category is None:
        client.annotate(_evidence(), [])
    else:
        with pytest.raises(AnnotationProviderError) as raised:
            client.annotate(_evidence(), [])
        assert raised.value.category == expected_category

    assert recorder.attempts == 1, "a non-truncation outcome must not be retried"


def test_retry_ceiling_must_exceed_the_base_ceiling(tmp_path: Path) -> None:
    """Retrying at the same ceiling repeats the truncation and bills for it twice."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": "bad-v1",
                "system_instructions": "test",
                "temperature": 0,
                "max_tokens": 500,
                "truncation_retry_max_tokens": 500,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must exceed max_tokens"):
        load_prompt_configuration(path)


# --- both attempts stay on the bill -------------------------------------------


def test_recovered_truncation_keeps_the_discarded_attempt_on_the_bill() -> None:
    recorder = _Recorder(
        _response(TRUNCATED_BODY, finish_reason="length", completion_tokens=500),
        _response(json.dumps(VALID_ANNOTATION), completion_tokens=300),
    )

    annotation = _client(recorder).annotate(_evidence(), [])

    # The accepted basis stays pure: it describes the call that produced this text.
    assert annotation.usage is not None
    assert annotation.usage.completion_tokens == 300
    # The discarded attempt is carried alongside rather than folded in or dropped.
    assert annotation.superseded_usage is not None
    assert annotation.superseded_usage.completion_tokens == 500
    assert annotation.superseded_usage.prompt_tokens == 700


def test_failed_retry_reports_both_attempts() -> None:
    recorder = _Recorder(
        _response(TRUNCATED_BODY, finish_reason="length", completion_tokens=500),
        _response("still not json", completion_tokens=120),
    )

    with pytest.raises(AnnotationProviderError) as raised:
        _client(recorder).annotate(_evidence(), [])

    error = raised.value
    assert error.category == "annotation_json_invalid"
    assert error.attempts == 2
    assert error.usage is not None and error.usage.completion_tokens == 120
    assert error.superseded_usage is not None
    assert error.superseded_usage.completion_tokens == 500


def test_report_totals_both_attempts_after_a_recovered_truncation() -> None:
    recorder = _Recorder(
        _response(TRUNCATED_BODY, finish_reason="length", completion_tokens=500),
        _response(json.dumps(VALID_ANNOTATION), completion_tokens=300),
    )

    report = run_investigation(
        _evidence(), retriever=_retriever(), annotator=_client(recorder)
    )

    assert report.status == "draft_requires_human_review"
    assert report.external_call_attempts == 2
    assert report.annotation_error_category is None
    assert report.unusable_call_usage is not None
    assert report.unusable_call_usage.completion_tokens == 500


def test_report_totals_both_attempts_when_the_retry_also_fails() -> None:
    recorder = _Recorder(
        _response(TRUNCATED_BODY, finish_reason="length", completion_tokens=500),
        _response(TRUNCATED_BODY, finish_reason="length", completion_tokens=900),
    )

    report = run_investigation(
        _evidence(), retriever=_retriever(), annotator=_client(recorder)
    )

    assert report.external_call_attempts == 2
    assert report.annotation_error_category == "annotation_truncated"
    assert report.unusable_call_usage is not None
    assert report.unusable_call_usage.prompt_tokens == 1400
    assert report.unusable_call_usage.completion_tokens == 1400


def test_fact_gate_rejection_after_a_retry_bills_both_attempts() -> None:
    rejected = {
        "evidence_references": ["model_probabilities.catboost"],
        "analytical_considerations": ["The score reached 0.83 this week."],
        "recommended_questions": ["What corroborates the pattern?"],
    }
    recorder = _Recorder(
        _response(TRUNCATED_BODY, finish_reason="length", completion_tokens=500),
        _response(json.dumps(rejected), completion_tokens=300),
    )

    report = run_investigation(
        _evidence(), retriever=_retriever(), annotator=_client(recorder)
    )

    assert report.status == "rejected_facts"
    assert report.llm_annotation is None
    # Neither the discarded attempt nor the rejected one may vanish from the bill.
    assert report.unusable_call_usage is not None
    assert report.unusable_call_usage.completion_tokens == 800
    assert report.unusable_call_usage.prompt_tokens == 1400


def test_partial_usage_across_attempts_reports_unknown_not_half() -> None:
    """One attempt without reported tokens makes the total unknowable, not smaller."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        if payload["max_tokens"] == BASE_MAX_TOKENS:
            return _response(TRUNCATED_BODY, finish_reason="length")
        # The retry succeeds but the provider omits usage entirely.
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(VALID_ANNOTATION)}}]}
        )

    annotator = _client(handler)

    report = run_investigation(_evidence(), retriever=_retriever(), annotator=annotator)

    assert report.llm_annotation is not None, "the retry produced a usable annotation"
    assert report.llm_annotation.usage is None
    assert report.unusable_call_usage is not None
    assert report.unusable_call_usage.prompt_tokens == 700

    summary = evaluate_golden_set(
        [_golden_case("retry-case-partial")],
        retriever=_retriever(),
        annotator=annotator,
    )

    # 700 known + one unreported call is not "700 billed"; the total is unknown, so the
    # case drops out of the billable basis and lowers coverage instead of understating.
    assert summary.external_call_total == 2
    assert summary.billable_call_count == 0
    assert summary.billable_token_coverage_rate == 0.0
    assert summary.billable_prompt_tokens == 0


# --- Golden aggregation --------------------------------------------------------


def _golden_case(case_id: str) -> GoldenCase:
    return GoldenCase(
        case_id=case_id,
        case_category="typology",
        evidence=_evidence(case_id),
    )


def test_golden_summary_counts_calls_separately_from_cases() -> None:
    recorder = _Recorder(
        _response(TRUNCATED_BODY, finish_reason="length", completion_tokens=500),
        _response(json.dumps(VALID_ANNOTATION), completion_tokens=300),
    )

    summary = evaluate_golden_set(
        [_golden_case("retry-case-01")],
        retriever=_retriever(),
        annotator=_client(recorder),
    )

    assert summary.external_case_count == 1
    assert summary.external_call_total == 2
    assert summary.truncation_retry_count == 1
    assert summary.truncation_retry_recovered_count == 1
    # The case succeeded, so the per-case parse rate is 1.0 and must be read next to
    # truncation_retry_count rather than as a first-attempt rate.
    assert summary.external_parse_success_rate == 1.0


def test_recovered_case_still_reports_the_wasted_first_attempt() -> None:
    recorder = _Recorder(
        _response(TRUNCATED_BODY, finish_reason="length", completion_tokens=500),
        _response(json.dumps(VALID_ANNOTATION), completion_tokens=300),
    )

    summary = evaluate_golden_set(
        [_golden_case("retry-case-02")],
        retriever=_retriever(),
        annotator=_client(recorder),
    )

    # Accepted basis: only the call that produced the annotation.
    assert summary.reported_prompt_tokens == 700
    assert summary.reported_completion_tokens == 300
    # Billable basis: both calls. Reading only the first would understate by half.
    assert summary.billable_prompt_tokens == 1400
    assert summary.billable_completion_tokens == 800
    # 1400 * 3.0 / 1e6 + 800 * 12.0 / 1e6
    assert summary.billable_estimated_cost_usd == pytest.approx(0.0138)
    # Waste survives recovery: the truncated attempt was paid for and thrown away.
    assert summary.wasted_prompt_tokens == 700
    assert summary.wasted_completion_tokens == 500
    assert summary.wasted_estimated_cost_usd == pytest.approx(0.0081)
    assert summary.wasted_token_share == pytest.approx(1200 / 2200)


def test_run_without_retries_keeps_its_previous_call_accounting() -> None:
    """The historical one-call-per-case reading must not shift under the new fields."""
    recorder = _Recorder(_response(json.dumps(VALID_ANNOTATION)))

    summary = evaluate_golden_set(
        [_golden_case("retry-case-03")],
        retriever=_retriever(),
        annotator=_client(recorder),
    )

    assert summary.external_case_count == 1
    assert summary.external_call_total == 1
    assert summary.truncation_retry_count == 0
    assert summary.wasted_prompt_tokens == 0
    assert summary.wasted_estimated_cost_usd == 0.0


# --- the v7 candidate ----------------------------------------------------------


def test_v7_changes_only_generation_limits() -> None:
    """Content metrics carry over from v6 only if the instructions are byte-identical."""
    v6 = yaml.safe_load(V6_PATH.read_text(encoding="utf-8"))
    v7 = yaml.safe_load(V7_PATH.read_text(encoding="utf-8"))

    assert v7["system_instructions"] == v6["system_instructions"]
    assert v7["temperature"] == v6["temperature"]
    assert v7["max_tokens"] == v6["max_tokens"]
    assert v7["version"] == "ecnu-risk-evidence-v7"
    assert v7["truncation_retry_max_tokens"] == RETRY_MAX_TOKENS


def test_v6_keeps_the_retry_disabled() -> None:
    """Holdout v3 measured v6 with no retry available; that must stay reproducible."""
    assert load_prompt_configuration(V6_PATH).truncation_retry_max_tokens is None


def test_v7_loads_with_the_retry_enabled() -> None:
    configuration = load_prompt_configuration(V7_PATH)

    assert configuration.version == "ecnu-risk-evidence-v7"
    assert configuration.max_tokens == BASE_MAX_TOKENS
    assert configuration.truncation_retry_max_tokens == RETRY_MAX_TOKENS


def test_shipped_default_still_points_at_v6() -> None:
    """v7 is a candidate; promoting it requires a preregistered holdout, not an edit."""
    assert DEFAULT_PROMPT_CONFIGURATION.version == "ecnu-risk-evidence-v6"
    assert DEFAULT_PROMPT_CONFIGURATION.truncation_retry_max_tokens is None
