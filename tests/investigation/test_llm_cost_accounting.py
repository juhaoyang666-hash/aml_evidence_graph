"""Cost accounting must cover billed calls, not only accepted annotations."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from aml_evidence_graph.evidence.package import RiskEvidencePackage
from aml_evidence_graph.evidence.typology import LocalBM25TypologyRetriever, TypologyDocument
from aml_evidence_graph.investigation.golden import GoldenCase, evaluate_golden_set
from aml_evidence_graph.investigation.llm import (
    AnnotationProviderError,
    ECNUAnnotationClient,
    PromptConfiguration,
)
from aml_evidence_graph.investigation.workflow import run_investigation

# Deliberately non-round prices so a missing /1_000_000 or a swapped input/output rate
# cannot coincidentally produce the expected total.
INPUT_PRICE = 3.0
OUTPUT_PRICE = 12.0


def _evidence() -> RiskEvidencePackage:
    return RiskEvidencePackage(
        alert_id="alert-cost-basis",
        generated_at=datetime(2026, 8, 4, tzinfo=UTC),
        transaction_id="txn-cost-basis",
        event_timestamp=datetime(2023, 7, 1, tzinfo=UTC),
        model_probabilities={"catboost": 0.8},
    )


def _retriever() -> LocalBM25TypologyRetriever:
    return LocalBM25TypologyRetriever(
        [
            TypologyDocument(
                typology_id="T-COST",
                version="1",
                title="Cost basis typology",
                source="test",
                body="Transaction risk investigation.",
            )
        ]
    )


# Pinned to a no-retry configuration on purpose. These tests are about what a single
# billed call contributes to each cost basis; inheriting the shipped default would make
# them change meaning whenever its generation limits change, as promoting v7 did.
# Retry-inclusive accounting is covered by tests/investigation/test_truncation_retry.py.
_NO_RETRY_PROMPT = PromptConfiguration(
    version="test-prompt-no-retry",
    system_instructions="test",
    temperature=0,
    max_tokens=500,
    truncation_retry_max_tokens=None,
)


def _client(handler) -> ECNUAnnotationClient:
    return ECNUAnnotationClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        input_cost_per_million_tokens_usd=INPUT_PRICE,
        output_cost_per_million_tokens_usd=OUTPUT_PRICE,
        prompt_configuration=_NO_RETRY_PROMPT,
    )


def _response(content: str, *, finish_reason: str | None = None) -> httpx.Response:
    choice: dict[str, object] = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return httpx.Response(
        200,
        json={
            "usage": {
                "prompt_tokens": 700,
                "completion_tokens": 300,
                "total_tokens": 1000,
            },
            "choices": [choice],
        },
    )


def _truncated_handler(request: httpx.Request) -> httpx.Response:
    del request
    return _response(
        '{"evidence_references": [], "analytical_considerations": [',
        finish_reason="length",
    )


@pytest.mark.parametrize(
    ("content", "finish_reason", "expected_category"),
    [
        ('{"evidence_references": [', "length", "annotation_truncated"),
        ("not json at all", None, "annotation_json_invalid"),
        ('["not", "an", "object"]', None, "annotation_schema_invalid"),
        (
            json.dumps({"evidence_references": {"unexpected": "mapping"}}),
            None,
            "annotation_schema_invalid",
        ),
    ],
)
def test_unparseable_response_still_reports_billed_tokens(
    content: str,
    finish_reason: str | None,
    expected_category: str,
) -> None:
    """The provider billed this call; the error must carry its usage, not drop it."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _response(content, finish_reason=finish_reason)

    with pytest.raises(AnnotationProviderError) as raised:
        _client(handler).annotate(_evidence(), [])

    error = raised.value
    assert error.category == expected_category
    assert error.usage is not None, "billed tokens were dropped on the failure path"
    assert error.usage.prompt_tokens == 700
    assert error.usage.completion_tokens == 300
    # 700 * 3.0 / 1e6 + 300 * 12.0 / 1e6
    assert error.usage.estimated_cost_usd == pytest.approx(0.0057)


def test_transport_failure_reports_no_usage_rather_than_guessing() -> None:
    """No response body means the billed amount is unknown and must stay null."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ConnectError("boom")

    with pytest.raises(AnnotationProviderError) as raised:
        _client(handler).annotate(_evidence(), [])

    assert raised.value.category == "transport_error"
    assert raised.value.usage is None


def test_report_surfaces_unusable_call_usage_for_a_truncated_call() -> None:
    report = run_investigation(
        _evidence(),
        retriever=_retriever(),
        annotator=_client(_truncated_handler),
    )

    assert report.llm_annotation is None
    assert report.annotation_error_category == "annotation_truncated"
    assert report.unusable_call_usage is not None
    assert report.unusable_call_usage.prompt_tokens == 700
    assert report.unusable_call_usage.estimated_cost_usd == pytest.approx(0.0057)


def test_fact_gate_rejection_keeps_the_call_on_the_billable_basis() -> None:
    """A rejected annotation is dropped from the report, but it was still charged."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _response(
            json.dumps(
                {
                    "evidence_references": ["model_probabilities.catboost"],
                    # A numeric value trips the fact gate on purpose.
                    "analytical_considerations": ["The score reached 0.83 this week."],
                    "recommended_questions": ["What corroborates the pattern?"],
                }
            )
        )

    report = run_investigation(
        _evidence(),
        retriever=_retriever(),
        annotator=_client(handler),
    )

    assert report.status == "rejected_facts"
    assert report.llm_annotation is None
    assert report.unusable_call_usage is not None
    assert report.unusable_call_usage.prompt_tokens == 700


def _golden_case(case_id: str) -> GoldenCase:
    return GoldenCase(
        case_id=case_id,
        case_category="typology",
        evidence=_evidence().model_copy(update={"alert_id": case_id}),
    )


def test_golden_summary_separates_billable_from_accepted_basis() -> None:
    """A run whose only external call failed must still report non-zero billable spend."""
    summary = evaluate_golden_set(
        [_golden_case("cost-case-01")],
        retriever=_retriever(),
        annotator=_client(_truncated_handler),
    )

    # Accepted basis stays empty: nothing was usable.
    assert summary.reported_prompt_tokens == 0
    assert summary.reported_completion_tokens == 0
    assert summary.llm_annotation_rate == 0.0

    # Billable basis records what the provider actually charged.
    assert summary.billable_call_count == 1
    assert summary.billable_token_coverage_rate == 1.0
    assert summary.billable_prompt_tokens == 700
    assert summary.billable_completion_tokens == 300
    assert summary.billable_estimated_cost_usd == pytest.approx(0.0057)

    # Every billed token was wasted, and the share must say so plainly.
    assert summary.wasted_prompt_tokens == 700
    assert summary.wasted_completion_tokens == 300
    assert summary.wasted_estimated_cost_usd == pytest.approx(0.0057)
    assert summary.wasted_token_share == 1.0


def test_accepted_annotation_is_billable_but_not_wasted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _response(
            json.dumps(
                {
                    "evidence_references": ["model_probabilities.catboost"],
                    "analytical_considerations": [
                        "Consider corroborating the observed pattern."
                    ],
                    "recommended_questions": [
                        "What contextual records could corroborate the pattern?"
                    ],
                }
            )
        )

    summary = evaluate_golden_set(
        [_golden_case("cost-case-02")],
        retriever=_retriever(),
        annotator=_client(handler),
    )

    assert summary.reported_prompt_tokens == 700
    assert summary.billable_prompt_tokens == 700
    assert summary.wasted_prompt_tokens == 0
    assert summary.wasted_token_share == 0.0
    # Nothing was wasted, so zero is knowable and must not be reported as null.
    assert summary.billable_estimated_cost_usd == pytest.approx(0.0057)
    assert summary.wasted_estimated_cost_usd == 0.0


def test_unpriced_run_keeps_cost_null_on_both_bases() -> None:
    """Without configured prices, tokens are still counted but cost stays unavailable."""
    unpriced = ECNUAnnotationClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(_truncated_handler)),
        prompt_configuration=_NO_RETRY_PROMPT,
    )
    summary = evaluate_golden_set(
        [_golden_case("cost-case-03")],
        retriever=_retriever(),
        annotator=unpriced,
    )

    # Tokens are counted, but an unpriced non-empty basis must stay null rather than
    # be understated as if unpriced calls were free.
    assert summary.billable_prompt_tokens == 700
    assert summary.billable_estimated_cost_usd is None
    assert summary.wasted_estimated_cost_usd is None
    assert summary.estimated_cost_usd is None


def test_template_only_run_reports_zero_billable_cost() -> None:
    """No external call at all means zero spend, which is knowable without prices."""
    summary = evaluate_golden_set([_golden_case("cost-case-04")], retriever=_retriever())

    assert summary.billable_call_count == 0
    assert summary.billable_token_coverage_rate is None
    assert summary.billable_prompt_tokens == 0
    assert summary.billable_estimated_cost_usd == 0.0
    assert summary.wasted_estimated_cost_usd == 0.0
    assert summary.wasted_token_share is None
