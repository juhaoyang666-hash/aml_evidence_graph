"""Operations rollup must separate failure modes and never treat missing usage as free."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aml_evidence_graph.evaluation.llm_operations import (
    build_operations_report,
    classify_error_category,
    load_run_summary,
    price_recorded_tokens,
    repeated_configuration_variance,
    summarize_run_operations,
)


def _case(
    *,
    attempted: bool = True,
    parsed: bool = True,
    fact_passed: bool | None = True,
    used: bool = True,
    error: str | None = None,
    latency_ms: float = 100.0,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    billable_prompt_tokens: int | None = None,
    estimated_cost_usd: float | None = None,
    billable_estimated_cost_usd: float | None = None,
) -> dict[str, object]:
    return {
        "external_call_attempted": attempted,
        "annotation_parse_succeeded": parsed,
        "fact_validation_passed": fact_passed,
        "annotation_used": used,
        "external_error_category": error,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "billable_prompt_tokens": billable_prompt_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "billable_estimated_cost_usd": billable_estimated_cost_usd,
    }


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("timeout", "availability"),
        ("http_status_500", "availability"),
        ("http_status_429", "availability"),
        ("transport_error", "availability"),
        ("response_json_invalid", "availability"),
        ("annotation_json_invalid", "format"),
        ("annotation_truncated", "format"),
        ("annotation_schema_invalid", "format"),
        ("external_annotation_error", "other"),
    ],
)
def test_error_categories_split_availability_from_format(category: str, expected: str) -> None:
    """A provider outage and a broken prompt contract must not share one bucket."""
    assert classify_error_category(category) == expected


def test_funnel_counts_each_stage_independently() -> None:
    summary = {
        "prompt_versions": ["v-test"],
        "cases": [
            _case(),
            # parsed but rejected by the fact gate
            _case(fact_passed=False, used=False),
            # never parsed
            _case(parsed=False, fact_passed=None, used=False, error="annotation_truncated"),
            # provider outage
            _case(parsed=False, fact_passed=None, used=False, error="timeout"),
            # deterministic probe: no external call
            _case(attempted=False, parsed=False, fact_passed=None, used=False),
        ],
    }

    run = summarize_run_operations(summary, run_label="funnel")

    assert run.case_count == 5
    assert run.external_attempted == 4
    assert run.parsed == 2
    assert run.fact_gate_passed == 1
    assert run.accepted == 1
    assert run.parse_success_rate == 0.5
    assert run.fact_gate_pass_rate == 0.5
    assert run.acceptance_rate == 0.25
    assert run.availability_failures == 1
    assert run.format_failures == 1
    assert run.error_categories == {"annotation_truncated": 1, "timeout": 1}


def test_calls_without_usage_are_reported_unrecoverable_not_free() -> None:
    """Historical runs lack usage for failed calls; that must be visible, not hidden."""
    summary = {
        "prompt_versions": ["v-legacy"],
        "cases": [
            _case(prompt_tokens=700, completion_tokens=300),
            _case(parsed=False, fact_passed=None, used=False, error="annotation_json_invalid"),
            _case(parsed=False, fact_passed=None, used=False, error="annotation_json_invalid"),
        ],
    }

    run = summarize_run_operations(summary, run_label="legacy")

    assert run.usage_observed_calls == 1
    assert run.unrecoverable_usage_calls == 2
    assert run.usage_coverage_rate == pytest.approx(1 / 3)
    assert run.billable_prompt_tokens == 700
    # No price configured, so cost must stay null rather than imply the failures were free.
    assert run.billable_estimated_cost_usd is None
    assert run.cost_status == "unpriced"


def test_priced_run_with_missing_usage_is_flagged_incomplete() -> None:
    summary = {
        "prompt_versions": ["v-priced"],
        "cases": [
            _case(
                prompt_tokens=700,
                completion_tokens=300,
                estimated_cost_usd=0.0057,
                billable_estimated_cost_usd=0.0057,
            ),
            _case(parsed=False, fact_passed=None, used=False, error="timeout"),
        ],
    }

    run = summarize_run_operations(summary, run_label="priced")

    assert run.billable_estimated_cost_usd == pytest.approx(0.0057)
    assert run.unrecoverable_usage_calls == 1
    assert run.cost_status == "priced_but_incomplete"


def test_fully_covered_priced_run_is_complete() -> None:
    summary = {
        "prompt_versions": ["v-priced"],
        "cases": [
            _case(
                prompt_tokens=700,
                completion_tokens=300,
                estimated_cost_usd=0.0057,
                billable_estimated_cost_usd=0.0057,
            ),
            _case(
                parsed=False,
                fact_passed=None,
                used=False,
                error="annotation_truncated",
                billable_prompt_tokens=700,
                billable_estimated_cost_usd=0.0021,
            ),
        ],
    }

    run = summarize_run_operations(summary, run_label="complete")

    assert run.usage_coverage_rate == 1.0
    assert run.unrecoverable_usage_calls == 0
    assert run.cost_status == "priced_and_complete"
    assert run.billable_estimated_cost_usd == pytest.approx(0.0078)
    # Accepted basis excludes the truncated call; billable basis includes it.
    assert run.accepted_prompt_tokens == 700
    assert run.billable_prompt_tokens == 1400


def test_template_only_run_reports_no_external_calls() -> None:
    summary = {"prompt_versions": [], "cases": [_case(attempted=False, used=False)]}

    run = summarize_run_operations(summary, run_label="template")

    assert run.external_attempted == 0
    assert run.parse_success_rate is None
    assert run.cost_status == "no_external_calls"


def test_variance_reports_spread_for_an_unchanged_configuration() -> None:
    """Two runs of one prompt on one set bound how much of a gain can be real."""
    def run_with(label: str, parsed_count: int):
        cases = [_case() for _ in range(parsed_count)]
        cases += [
            _case(parsed=False, fact_passed=None, used=False, error="annotation_json_invalid")
            for _ in range(20 - parsed_count)
        ]
        return summarize_run_operations(
            {"prompt_versions": ["v-same"], "cases": cases}, run_label=label
        )

    runs = [run_with("run_a", 19), run_with("run_b", 15)]

    variance = repeated_configuration_variance(runs)

    assert len(variance) == 1
    item = variance[0]
    assert item["prompt_version"] == "v-same"
    assert item["run_count"] == 2
    assert item["parse_success_rate_min"] == pytest.approx(0.75)
    assert item["parse_success_rate_max"] == pytest.approx(0.95)
    assert item["parse_success_rate_spread"] == pytest.approx(0.20)


def test_variance_ignores_single_runs_and_mixed_prompt_versions() -> None:
    single = summarize_run_operations(
        {"prompt_versions": ["v-one"], "cases": [_case()]}, run_label="single"
    )
    mixed = summarize_run_operations(
        {"prompt_versions": ["v-a", "v-b"], "cases": [_case()]}, run_label="mixed"
    )

    assert repeated_configuration_variance([single, mixed]) == []


def test_report_totals_and_limitations_are_present() -> None:
    runs = [
        summarize_run_operations(
            {
                "prompt_versions": ["v-x"],
                "cases": [
                    _case(prompt_tokens=10, completion_tokens=5),
                    _case(parsed=False, fact_passed=None, used=False, error="timeout"),
                ],
            },
            run_label="only",
        )
    ]

    report = build_operations_report(runs, report_id="r-1", generated_at="2026-08-04T00:00:00Z")

    assert report["external_calls_total"] == 2
    assert report["usage_observed_calls_total"] == 1
    assert report["unrecoverable_usage_calls_total"] == 1
    assert report["usage_coverage_rate_total"] == 0.5
    assert len(report["runs"]) == 1
    assert any("unrecoverable" in item for item in report["limitations"])
    # The report must be publishable: no model text anywhere in it.
    assert "analytical_considerations" not in json.dumps(report)


def test_incomplete_coverage_never_reports_zero_waste() -> None:
    """A v4-shaped run (most calls failed, usage only for the survivors) must not read
    as "nothing was wasted" merely because the failures were never measured."""
    cases = [_case(prompt_tokens=700, completion_tokens=300)]
    cases += [
        _case(parsed=False, fact_passed=None, used=False, error="annotation_json_invalid")
        for _ in range(18)
    ]
    run = summarize_run_operations({"prompt_versions": ["v4"], "cases": cases}, run_label="v4")

    priced = price_recorded_tokens(
        run,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=3.0,
    )

    assert priced["unmeasured_calls"] == 18
    assert priced["is_complete_bill"] is False
    assert priced["billable_cost_is_lower_bound"] is True
    assert priced["wasted_cost_usd_from_recorded_tokens"] is None
    # 700 * 1.0 / 1e6 + 300 * 3.0 / 1e6, a floor rather than a total.
    assert priced["billable_cost_usd_from_recorded_tokens"] == pytest.approx(0.0016)


def test_complete_coverage_reports_exact_waste() -> None:
    run = summarize_run_operations(
        {
            "prompt_versions": ["v-complete"],
            "cases": [
                _case(prompt_tokens=700, completion_tokens=300),
                _case(
                    parsed=False,
                    fact_passed=None,
                    used=False,
                    error="annotation_truncated",
                    billable_prompt_tokens=500,
                ),
            ],
        },
        run_label="complete",
    )

    priced = price_recorded_tokens(
        run,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=3.0,
    )

    assert priced["is_complete_bill"] is True
    assert priced["billable_cost_is_lower_bound"] is False
    # Billable 1200 prompt / 300 completion; accepted 700 / 300.
    assert priced["billable_cost_usd_from_recorded_tokens"] == pytest.approx(0.0021)
    assert priced["accepted_cost_usd_from_recorded_tokens"] == pytest.approx(0.0016)
    assert priced["wasted_cost_usd_from_recorded_tokens"] == pytest.approx(0.0005)


def test_negative_prices_are_rejected() -> None:
    run = summarize_run_operations({"prompt_versions": ["v"], "cases": [_case()]}, run_label="r")

    with pytest.raises(ValueError, match="non-negative"):
        price_recorded_tokens(
            run,
            input_cost_per_million_tokens_usd=-1.0,
            output_cost_per_million_tokens_usd=3.0,
        )


def test_report_omits_pricing_when_no_price_is_configured() -> None:
    runs = [summarize_run_operations({"prompt_versions": ["v"], "cases": [_case()]}, run_label="r")]

    report = build_operations_report(runs, report_id="r", generated_at="2026-08-04T00:00:00Z")

    assert report["post_hoc_pricing"] is None


def test_load_run_summary_picks_up_sibling_manifest_run_id(tmp_path: Path) -> None:
    summary_path = tmp_path / "blind_run_summary.json"
    summary_path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    (tmp_path / "blind_run_summary_run_manifest.json").write_text(
        json.dumps({"run_id": "20260804T081410Z-d3eebce570"}), encoding="utf-8"
    )

    summary, run_id = load_run_summary(summary_path)

    assert summary == {"cases": []}
    assert run_id == "20260804T081410Z-d3eebce570"


def test_load_run_summary_tolerates_a_missing_manifest(tmp_path: Path) -> None:
    summary_path = tmp_path / "run.json"
    summary_path.write_text(json.dumps({"cases": []}), encoding="utf-8")

    _, run_id = load_run_summary(summary_path)

    assert run_id is None
