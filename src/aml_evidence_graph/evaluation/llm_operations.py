"""Operational monitoring and cost rollup over frozen external-LLM Golden runs.

This module only reads run summaries that already exist on disk. It performs no
external calls and publishes no model text, so it is safe to run in CI and to
export as public evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# A call that never produced a parseable body failed for one of two very different
# reasons, and conflating them hides whether a prompt or the provider is at fault.
AVAILABILITY_ERROR_PREFIXES: tuple[str, ...] = (
    "timeout",
    "http_status_",
    "transport_error",
    "response_json_invalid",
)
FORMAT_ERROR_CATEGORIES: frozenset[str] = frozenset(
    {
        "annotation_json_invalid",
        "annotation_truncated",
        "annotation_schema_invalid",
    }
)


def classify_error_category(category: str) -> str:
    """Split provider availability failures from prompt/format contract failures."""
    if category in FORMAT_ERROR_CATEGORIES:
        return "format"
    if category.startswith(AVAILABILITY_ERROR_PREFIXES):
        return "availability"
    return "other"


@dataclass(frozen=True)
class LLMRunOperations:
    """Funnel, failure-mode, latency and cost view of one frozen run."""

    run_label: str
    run_id: str | None
    prompt_versions: tuple[str, ...]
    case_count: int
    external_attempted: int
    parsed: int
    fact_gate_passed: int
    accepted: int
    parse_success_rate: float | None
    fact_gate_pass_rate: float | None
    acceptance_rate: float | None
    # Calls, not cases. Runs predating the truncation retry bill one call per case, so
    # these default to the case count rather than to zero when the field is absent.
    external_calls: int = 0
    truncation_retries: int = 0
    truncation_retries_recovered: int = 0
    error_categories: dict[str, int] = field(default_factory=dict)
    availability_failures: int = 0
    format_failures: int = 0
    other_failures: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_max_ms: float | None = None
    accepted_prompt_tokens: int = 0
    accepted_completion_tokens: int = 0
    accepted_estimated_cost_usd: float | None = None
    billable_prompt_tokens: int = 0
    billable_completion_tokens: int = 0
    billable_estimated_cost_usd: float | None = None
    usage_observed_calls: int = 0
    usage_coverage_rate: float | None = None
    unrecoverable_usage_calls: int = 0
    cost_status: str = "unpriced"


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_run_operations(
    summary: dict[str, object],
    *,
    run_label: str,
    run_id: str | None = None,
) -> LLMRunOperations:
    """Derive an operations view from one Golden run summary payload.

    Older summaries predate billable-basis accounting, so per-case usage is read
    defensively and calls with no recorded usage are counted as unrecoverable
    rather than silently treated as free.
    """
    cases = summary.get("cases")
    cases = cases if isinstance(cases, list) else []
    external = [
        case
        for case in cases
        if isinstance(case, dict) and case.get("external_call_attempted")
    ]
    parsed = [case for case in external if case.get("annotation_parse_succeeded")]
    fact_passed = [case for case in parsed if case.get("fact_validation_passed") is True]
    accepted = [case for case in external if case.get("annotation_used")]

    raw_errors = Counter(
        str(case["external_error_category"])
        for case in external
        if case.get("external_error_category")
    )
    grouped: Counter[str] = Counter()
    for category, count in raw_errors.items():
        grouped[classify_error_category(category)] += count

    latencies = sorted(
        float(case["latency_ms"])
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("latency_ms"), (int, float))
    )

    def usage_of(case: dict[str, object], field_name: str) -> int | float | None:
        for key in (f"billable_{field_name}", field_name):
            value = case.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        return None

    usage_calls = [
        case
        for case in external
        if usage_of(case, "prompt_tokens") is not None
        or usage_of(case, "completion_tokens") is not None
    ]
    billable_prompt = sum(int(usage_of(case, "prompt_tokens") or 0) for case in usage_calls)
    billable_completion = sum(
        int(usage_of(case, "completion_tokens") or 0) for case in usage_calls
    )
    accepted_prompt = sum(
        int(case.get("prompt_tokens") or 0)
        for case in accepted
        if isinstance(case.get("prompt_tokens"), int)
    )
    accepted_completion = sum(
        int(case.get("completion_tokens") or 0)
        for case in accepted
        if isinstance(case.get("completion_tokens"), int)
    )

    def attempts_of(case: dict[str, object]) -> int:
        # Absent means the run predates the retry, when a case was exactly one call.
        value = case.get("external_call_attempts")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return 1

    retried = [case for case in external if attempts_of(case) > 1]

    accepted_costs = [case.get("estimated_cost_usd") for case in accepted]
    billable_costs = [
        usage_of(case, "estimated_cost_usd") for case in usage_calls
    ]
    priced = bool(billable_costs) and all(value is not None for value in billable_costs)
    unrecoverable = len(external) - len(usage_calls)
    if not external:
        cost_status = "no_external_calls"
    elif not priced:
        cost_status = "unpriced"
    elif unrecoverable:
        cost_status = "priced_but_incomplete"
    else:
        cost_status = "priced_and_complete"

    return LLMRunOperations(
        run_label=run_label,
        run_id=run_id,
        prompt_versions=tuple(
            str(version) for version in (summary.get("prompt_versions") or [])
        ),
        case_count=len(cases),
        external_attempted=len(external),
        parsed=len(parsed),
        fact_gate_passed=len(fact_passed),
        accepted=len(accepted),
        parse_success_rate=_ratio(len(parsed), len(external)),
        fact_gate_pass_rate=_ratio(len(fact_passed), len(parsed)),
        acceptance_rate=_ratio(len(accepted), len(external)),
        external_calls=sum(attempts_of(case) for case in external),
        truncation_retries=len(retried),
        truncation_retries_recovered=sum(1 for case in retried if case.get("annotation_used")),
        error_categories=dict(sorted(raw_errors.items())),
        availability_failures=grouped.get("availability", 0),
        format_failures=grouped.get("format", 0),
        other_failures=grouped.get("other", 0),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        latency_max_ms=latencies[-1] if latencies else None,
        accepted_prompt_tokens=accepted_prompt,
        accepted_completion_tokens=accepted_completion,
        accepted_estimated_cost_usd=(
            sum(float(value) for value in accepted_costs if value is not None)
            if accepted_costs and all(value is not None for value in accepted_costs)
            else None
        ),
        billable_prompt_tokens=billable_prompt,
        billable_completion_tokens=billable_completion,
        billable_estimated_cost_usd=(
            sum(float(value) for value in billable_costs if value is not None)
            if priced
            else None
        ),
        usage_observed_calls=len(usage_calls),
        usage_coverage_rate=_ratio(len(usage_calls), len(external)),
        unrecoverable_usage_calls=unrecoverable,
        cost_status=cost_status,
    )


def repeated_configuration_variance(
    runs: list[LLMRunOperations],
) -> list[dict[str, object]]:
    """Report parse-rate spread across runs that share prompt version and set size.

    The promotion chain gates on parse success rate, so run-to-run spread for an
    unchanged configuration bounds how much of any observed gain is real.
    """
    groups: dict[tuple[str, int], list[LLMRunOperations]] = {}
    for run in runs:
        if run.external_attempted == 0 or len(run.prompt_versions) != 1:
            continue
        groups.setdefault((run.prompt_versions[0], run.external_attempted), []).append(run)
    variance: list[dict[str, object]] = []
    for (prompt_version, attempted), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        rates = [run.parse_success_rate for run in members if run.parse_success_rate is not None]
        if len(rates) < 2:
            continue
        variance.append(
            {
                "prompt_version": prompt_version,
                "external_attempted": attempted,
                "run_count": len(members),
                "run_labels": [run.run_label for run in members],
                "parse_success_rate_min": min(rates),
                "parse_success_rate_max": max(rates),
                "parse_success_rate_spread": max(rates) - min(rates),
            }
        )
    return variance


def price_recorded_tokens(
    run: LLMRunOperations,
    *,
    input_cost_per_million_tokens_usd: float,
    output_cost_per_million_tokens_usd: float,
) -> dict[str, object]:
    """Apply a contract price to already-recorded tokens.

    Price is a reporting parameter, not a capture parameter: tokens must be captured at
    call time and cannot be reconstructed, whereas a price can be applied to any run
    afterwards. Coverage travels with the figure so a run missing usage for failed calls
    is never presented as a complete bill.
    """
    if input_cost_per_million_tokens_usd < 0 or output_cost_per_million_tokens_usd < 0:
        raise ValueError("Token prices must be non-negative.")
    billable = (
        run.billable_prompt_tokens * input_cost_per_million_tokens_usd
        + run.billable_completion_tokens * output_cost_per_million_tokens_usd
    ) / 1_000_000
    accepted = (
        run.accepted_prompt_tokens * input_cost_per_million_tokens_usd
        + run.accepted_completion_tokens * output_cost_per_million_tokens_usd
    ) / 1_000_000
    complete = run.unrecoverable_usage_calls == 0 and run.external_attempted > 0
    return {
        "run_label": run.run_label,
        "input_cost_per_million_tokens_usd": input_cost_per_million_tokens_usd,
        "output_cost_per_million_tokens_usd": output_cost_per_million_tokens_usd,
        "billable_cost_usd_from_recorded_tokens": billable,
        # Incomplete coverage makes the billable figure a floor, never a total.
        "billable_cost_is_lower_bound": not complete,
        "accepted_cost_usd_from_recorded_tokens": accepted,
        # With calls missing usage, the difference collapses to 0.0, which would read as
        # "nothing was wasted" when the truth is "waste was never measured".
        "wasted_cost_usd_from_recorded_tokens": (billable - accepted) if complete else None,
        "unmeasured_calls": run.unrecoverable_usage_calls,
        "usage_coverage_rate": run.usage_coverage_rate,
        "unrecoverable_usage_calls": run.unrecoverable_usage_calls,
        "is_complete_bill": complete,
    }


def build_operations_report(
    runs: list[LLMRunOperations],
    *,
    report_id: str,
    generated_at: str,
    input_cost_per_million_tokens_usd: float | None = None,
    output_cost_per_million_tokens_usd: float | None = None,
) -> dict[str, object]:
    """Assemble a publishable operations report with no model text or evidence bodies."""
    external_total = sum(run.external_attempted for run in runs)
    call_total = sum(run.external_calls for run in runs)
    usage_total = sum(run.usage_observed_calls for run in runs)
    pricing: list[dict[str, object]] | None = None
    if (
        input_cost_per_million_tokens_usd is not None
        and output_cost_per_million_tokens_usd is not None
    ):
        pricing = [
            price_recorded_tokens(
                run,
                input_cost_per_million_tokens_usd=input_cost_per_million_tokens_usd,
                output_cost_per_million_tokens_usd=output_cost_per_million_tokens_usd,
            )
            for run in runs
        ]
    return {
        "post_hoc_pricing": pricing,
        "schema_version": "1.0",
        "report_id": report_id,
        "generated_at": generated_at,
        "scope": (
            "Operational monitoring and cost rollup over frozen local Golden runs. "
            "Not a provider SLA and not an independent evaluation."
        ),
        "external_cases_total": external_total,
        "external_calls_total": call_total,
        "truncation_retries_total": sum(run.truncation_retries for run in runs),
        "truncation_retries_recovered_total": sum(
            run.truncation_retries_recovered for run in runs
        ),
        "usage_observed_calls_total": usage_total,
        "usage_coverage_rate_total": _ratio(usage_total, external_total),
        "unrecoverable_usage_calls_total": external_total - usage_total,
        "runs": [asdict(run) for run in runs],
        "repeated_configuration_variance": repeated_configuration_variance(runs),
        "limitations": [
            "Token usage for calls that failed parsing was not captured before the "
            "billable-basis fix, so historical spend for those calls is unrecoverable.",
            "Monetary cost stays null unless contract prices are configured.",
            "Latency covers whole cases on a single local machine, including "
            "deterministic template work, and is not an online serving SLA.",
            "error_categories lists terminal failures only. A truncation that a retry "
            "recovered appears in truncation_retries, not in format_failures, so a zero "
            "format-failure count does not mean no truncation occurred.",
            "parse_success_rate is per case. Where truncation_retries is non-zero it is "
            "not a first-attempt rate, and usage_coverage_rate stays on the case basis "
            "so its denominator matches.",
        ],
    }


def load_run_summary(path: Path) -> tuple[dict[str, object], str | None]:
    """Load a run summary and its sibling run manifest run_id when present."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"Run summary must be a JSON object: {path}")
    manifest_path = path.with_name(f"{path.stem}_run_manifest.json")
    run_id: str | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            value = manifest.get("run_id")
            run_id = str(value) if value is not None else None
    return summary, run_id
