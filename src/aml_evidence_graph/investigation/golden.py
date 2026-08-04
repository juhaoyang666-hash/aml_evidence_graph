"""Golden-set evaluation for deterministic and LLM-assisted investigation drafts."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aml_evidence_graph.evidence.package import (
    InvestigationAnnotation,
    InvestigationReport,
    RiskEvidencePackage,
)
from aml_evidence_graph.evidence.typology import LocalBM25TypologyRetriever, load_typology_documents
from aml_evidence_graph.investigation.evaluation import evaluate_investigation_report
from aml_evidence_graph.investigation.llm import EvidenceAnnotationClient
from aml_evidence_graph.investigation.workflow import run_investigation_traced
from aml_evidence_graph.settings import Settings
from aml_evidence_graph.tracking.run import create_run_manifest


class GoldenCase(BaseModel):
    """A deidentified expected-behavior case; not a model-training record."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    evidence: RiskEvidencePackage
    expected_typology_ids: list[str] = Field(default_factory=list)
    expect_rejected_facts: bool = False
    case_category: Literal["typology", "low_evidence", "adversarial"] | None = None
    # When set, forces this annotation through validation (hallucination-intercept probes).
    injected_annotation: InvestigationAnnotation | None = None


class _FixedAnnotationClient:
    """Deterministic annotator used for hallucination-intercept Golden probes."""

    def __init__(self, annotation: InvestigationAnnotation) -> None:
        self._annotation = annotation

    def annotate(
        self,
        evidence: RiskEvidencePackage,
        references: list[object],
    ) -> InvestigationAnnotation:
        del evidence, references
        return self._annotation


@dataclass(frozen=True)
class GoldenCaseResult:
    """Per-case test evidence suitable for prompt-regression comparison."""

    case_id: str
    case_category: str | None
    schema_valid: bool
    fact_snapshot_matches: bool
    evidence_coverage: float
    typology_match: bool | None
    correct_rejection: bool
    no_evidence_refusal: bool
    hallucination_intercepted: bool | None
    tool_call_limit_respected: bool
    latency_ms: float
    external_call_attempted: bool
    annotation_parse_succeeded: bool
    fact_validation_passed: bool | None
    external_error_category: str | None
    annotation_used: bool
    report_status: str
    prompt_version: str | None
    model_name: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost_usd: float | None
    billable_prompt_tokens: int | None
    billable_completion_tokens: int | None
    billable_estimated_cost_usd: float | None
    evidence_references: list[str]
    analytical_considerations: list[str]
    recommended_questions: list[str]
    fact_validation_errors: list[str]


@dataclass(frozen=True)
class GoldenSetSummary:
    """Aggregate quality gates for a single prompt/model/knowledge version."""

    case_count: int
    schema_compliance_rate: float
    fact_snapshot_match_rate: float
    mean_evidence_coverage: float
    typology_match_rate: float | None
    correct_rejection_rate: float
    no_evidence_refusal_rate: float | None
    hallucination_intercept_rate: float | None
    tool_limit_pass_rate: float
    mean_latency_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    external_case_count: int
    external_parse_success_rate: float | None
    external_fact_validation_pass_rate: float | None
    llm_annotation_rate: float
    token_usage_coverage_rate: float
    reported_prompt_tokens: int
    reported_completion_tokens: int
    estimated_cost_usd: float | None
    # Billable basis: every attempted external call whose usage the provider reported,
    # including calls whose output failed parsing or the fact gate. The `reported_*`
    # fields above stay on the accepted-annotation basis so published numbers keep
    # their original meaning.
    billable_call_count: int
    billable_token_coverage_rate: float | None
    billable_prompt_tokens: int
    billable_completion_tokens: int
    billable_estimated_cost_usd: float | None
    wasted_prompt_tokens: int
    wasted_completion_tokens: int
    wasted_estimated_cost_usd: float | None
    wasted_token_share: float | None
    prompt_versions: list[str]
    model_names: list[str]
    category_counts: dict[str, int]
    cases: list[GoldenCaseResult]


def load_golden_cases(path: Path) -> list[GoldenCase]:
    """Load deidentified Golden Case JSON from the local repository."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [GoldenCase.model_validate(payload)]
    if isinstance(payload, list):
        return [GoldenCase.model_validate(item) for item in payload]
    raise ValueError("Golden case file must contain a JSON object or array.")


def _sum_cost(values: list[float | None]) -> float | None:
    """Total a cost column, distinguishing "exactly zero" from "not computable".

    An empty column means no call fell in this basis, so the cost is exactly zero
    regardless of whether prices are configured. A populated column with any null
    means prices were unavailable, and the total must stay null rather than be
    silently understated by treating unpriced calls as free.
    """
    if not values:
        return 0.0
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _billable_usage_value(
    report: InvestigationReport,
    field: str,
) -> int | float | None:
    """Read one usage field on the billable basis: accepted or billed-but-unusable.

    Exactly one of the two carries usage for a given case, because a report either
    exposes an accepted annotation or records the unusable call that replaced it.
    """
    for usage in (
        report.llm_annotation.usage if report.llm_annotation is not None else None,
        report.unusable_call_usage,
    ):
        if usage is None:
            continue
        value = getattr(usage, field)
        if value is not None:
            return value
    return None


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def evaluate_golden_set(
    cases: list[GoldenCase],
    *,
    retriever: LocalBM25TypologyRetriever,
    annotator: EvidenceAnnotationClient | None = None,
    trace_sink: list[dict[str, object]] | None = None,
) -> GoldenSetSummary:
    """Evaluate schema, factual snapshot, evidence coverage, retrieval and latency.

    When ``trace_sink`` is supplied, one correlated node trace per case is appended to
    it. Traces stay out of the summary so run summaries do not grow without bound.
    """
    if not cases:
        raise ValueError("At least one Golden Case is required.")
    results: list[GoldenCaseResult] = []
    for case in cases:
        case_annotator: EvidenceAnnotationClient | _FixedAnnotationClient | None
        if case.injected_annotation is not None:
            case_annotator = _FixedAnnotationClient(case.injected_annotation)
        else:
            case_annotator = annotator
        started_at = time.perf_counter()
        report, trace = run_investigation_traced(
            case.evidence,
            retriever=retriever,
            annotator=case_annotator,
            trace_id=f"golden-{case.case_id}",
        )
        if trace_sink is not None:
            trace_sink.append({**trace, "case_id": case.case_id})
        latency_ms = (time.perf_counter() - started_at) * 1_000
        expected_snapshot = case.evidence.model_dump(mode="json")
        evidence_items = (
            len(case.evidence.model_probabilities)
            + len(case.evidence.rule_hits)
            + len(case.evidence.key_features)
        )
        coverage = (
            min(1.0, len(report.factual_summary) / evidence_items)
            if evidence_items
            else 0.0
        )
        observed_typologies = {
            reference.typology_id
            for reference in case.evidence.typology_references
        }
        observed_typologies.update(
            item.split(" v", maxsplit=1)[0].removeprefix("Consider ")
            for item in report.typology_considerations
            if item.startswith("Consider ")
        )
        typology_match = (
            bool(set(case.expected_typology_ids).intersection(observed_typologies))
            if case.expected_typology_ids
            else None
        )
        gate = evaluate_investigation_report(case.evidence, report)
        external_call_attempted = annotator is not None and case.injected_annotation is None
        validation_errors = (
            list(report.fact_validation.errors)
            if report.fact_validation is not None
            else []
        )
        annotation_parse_succeeded = external_call_attempted and (
            report.llm_annotation is not None or bool(validation_errors)
        )
        fact_validation_passed = (
            report.fact_validation.valid
            if annotation_parse_succeeded and report.fact_validation is not None
            else None
        )
        exposed_annotation = report.llm_annotation
        hallucination_intercepted: bool | None = None
        if case.injected_annotation is not None and case.expect_rejected_facts:
            hallucination_intercepted = report.status == "rejected_facts"
        results.append(
            GoldenCaseResult(
                case_id=case.case_id,
                case_category=case.case_category,
                schema_valid=report.report_schema_version == "1.0",
                fact_snapshot_matches=report.fact_snapshot == expected_snapshot,
                evidence_coverage=coverage,
                typology_match=typology_match,
                correct_rejection=(
                    report.status == "rejected_facts"
                    if case.expect_rejected_facts
                    else report.status == "draft_requires_human_review"
                ),
                no_evidence_refusal=gate.no_evidence_refusal,
                hallucination_intercepted=hallucination_intercepted,
                tool_call_limit_respected=report.tool_call_count <= 4,
                latency_ms=latency_ms,
                external_call_attempted=external_call_attempted,
                annotation_parse_succeeded=annotation_parse_succeeded,
                fact_validation_passed=fact_validation_passed,
                external_error_category=report.annotation_error_category,
                annotation_used=report.llm_annotation is not None,
                report_status=report.status,
                prompt_version=(
                    report.llm_annotation.prompt_version
                    if report.llm_annotation is not None
                    else None
                ),
                model_name=(
                    report.llm_annotation.model_name
                    if report.llm_annotation is not None
                    else None
                ),
                prompt_tokens=(
                    report.llm_annotation.usage.prompt_tokens
                    if report.llm_annotation is not None
                    and report.llm_annotation.usage is not None
                    else None
                ),
                completion_tokens=(
                    report.llm_annotation.usage.completion_tokens
                    if report.llm_annotation is not None
                    and report.llm_annotation.usage is not None
                    else None
                ),
                estimated_cost_usd=(
                    report.llm_annotation.usage.estimated_cost_usd
                    if report.llm_annotation is not None
                    and report.llm_annotation.usage is not None
                    else None
                ),
                billable_prompt_tokens=_billable_usage_value(report, "prompt_tokens"),
                billable_completion_tokens=_billable_usage_value(
                    report, "completion_tokens"
                ),
                billable_estimated_cost_usd=_billable_usage_value(
                    report, "estimated_cost_usd"
                ),
                evidence_references=(
                    list(exposed_annotation.evidence_references)
                    if exposed_annotation is not None
                    else []
                ),
                analytical_considerations=(
                    list(exposed_annotation.analytical_considerations)
                    if exposed_annotation is not None
                    else []
                ),
                recommended_questions=(
                    list(exposed_annotation.recommended_questions)
                    if exposed_annotation is not None
                    else []
                ),
                fact_validation_errors=validation_errors,
            )
        )
    comparable = [result.typology_match for result in results if result.typology_match is not None]
    annotations = [result for result in results if result.annotation_used]
    external_results = [result for result in results if result.external_call_attempted]
    parsed_external_results = [
        result for result in external_results if result.annotation_parse_succeeded
    ]
    token_usage = [
        result
        for result in annotations
        if result.prompt_tokens is not None or result.completion_tokens is not None
    ]
    cost_values = [result.estimated_cost_usd for result in annotations]
    # Billable basis covers every attempted call with provider-reported usage, so a run
    # whose annotations mostly failed still shows what the provider actually charged.
    billable_results = [
        result
        for result in external_results
        if result.billable_prompt_tokens is not None
        or result.billable_completion_tokens is not None
    ]
    billable_cost_values = [result.billable_estimated_cost_usd for result in billable_results]
    wasted_results = [result for result in billable_results if not result.annotation_used]
    wasted_cost_values = [result.billable_estimated_cost_usd for result in wasted_results]
    billable_prompt_tokens = sum(result.billable_prompt_tokens or 0 for result in billable_results)
    billable_completion_tokens = sum(
        result.billable_completion_tokens or 0 for result in billable_results
    )
    wasted_prompt_tokens = sum(result.billable_prompt_tokens or 0 for result in wasted_results)
    wasted_completion_tokens = sum(
        result.billable_completion_tokens or 0 for result in wasted_results
    )
    billable_total_tokens = billable_prompt_tokens + billable_completion_tokens
    wasted_total_tokens = wasted_prompt_tokens + wasted_completion_tokens
    prompt_versions = sorted(
        {result.prompt_version for result in annotations if result.prompt_version is not None}
    )
    model_names = sorted(
        {result.model_name for result in annotations if result.model_name is not None}
    )
    latencies = sorted(result.latency_ms for result in results)
    low_evidence_cases = [
        result
        for result, case in zip(results, cases, strict=True)
        if case.case_category == "low_evidence" or bool(case.evidence.missing_evidence)
    ]
    hallucination_probes = [
        result.hallucination_intercepted
        for result in results
        if result.hallucination_intercepted is not None
    ]
    category_counts: dict[str, int] = {}
    for result in results:
        key = result.case_category or "unspecified"
        category_counts[key] = category_counts.get(key, 0) + 1
    return GoldenSetSummary(
        case_count=len(results),
        schema_compliance_rate=sum(result.schema_valid for result in results) / len(results),
        fact_snapshot_match_rate=sum(
            result.fact_snapshot_matches for result in results
        )
        / len(results),
        mean_evidence_coverage=sum(result.evidence_coverage for result in results)
        / len(results),
        typology_match_rate=(sum(comparable) / len(comparable) if comparable else None),
        correct_rejection_rate=sum(result.correct_rejection for result in results)
        / len(results),
        no_evidence_refusal_rate=(
            sum(result.no_evidence_refusal for result in low_evidence_cases)
            / len(low_evidence_cases)
            if low_evidence_cases
            else None
        ),
        hallucination_intercept_rate=(
            sum(hallucination_probes) / len(hallucination_probes)
            if hallucination_probes
            else None
        ),
        tool_limit_pass_rate=sum(
            result.tool_call_limit_respected for result in results
        )
        / len(results),
        mean_latency_ms=sum(result.latency_ms for result in results) / len(results),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        external_case_count=len(external_results),
        external_parse_success_rate=(
            sum(result.annotation_parse_succeeded for result in external_results)
            / len(external_results)
            if external_results
            else None
        ),
        external_fact_validation_pass_rate=(
            sum(result.fact_validation_passed is True for result in parsed_external_results)
            / len(parsed_external_results)
            if parsed_external_results
            else None
        ),
        llm_annotation_rate=len(annotations) / len(results),
        token_usage_coverage_rate=(
            len(token_usage) / len(annotations) if annotations else 0.0
        ),
        reported_prompt_tokens=sum(result.prompt_tokens or 0 for result in token_usage),
        reported_completion_tokens=sum(
            result.completion_tokens or 0 for result in token_usage
        ),
        estimated_cost_usd=(
            sum(value for value in cost_values if value is not None)
            if cost_values and all(value is not None for value in cost_values)
            else None
        ),
        billable_call_count=len(billable_results),
        billable_token_coverage_rate=(
            len(billable_results) / len(external_results) if external_results else None
        ),
        billable_prompt_tokens=billable_prompt_tokens,
        billable_completion_tokens=billable_completion_tokens,
        billable_estimated_cost_usd=_sum_cost(billable_cost_values),
        wasted_prompt_tokens=wasted_prompt_tokens,
        wasted_completion_tokens=wasted_completion_tokens,
        wasted_estimated_cost_usd=_sum_cost(wasted_cost_values),
        wasted_token_share=(
            wasted_total_tokens / billable_total_tokens if billable_total_tokens else None
        ),
        prompt_versions=prompt_versions,
        model_names=model_names,
        category_counts=category_counts,
        cases=results,
    )


def golden_summary_as_dict(summary: GoldenSetSummary) -> dict[str, object]:
    """Serialize a Golden Set report without storing input evidence rows."""
    return asdict(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--typologies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Evaluate only the named case; repeat to select multiple cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    annotator = None
    if args.use_llm:
        from aml_evidence_graph.investigation.llm import ECNUAnnotationClient

        annotator = ECNUAnnotationClient.from_settings(settings)
    cases = load_golden_cases(args.cases)
    if args.case_id:
        selected_ids = set(args.case_id)
        cases = [case for case in cases if case.case_id in selected_ids]
        missing_ids = sorted(selected_ids.difference(case.case_id for case in cases))
        if missing_ids:
            raise ValueError("Unknown Golden case IDs: " + ", ".join(missing_ids))
    traces: list[dict[str, object]] = []
    summary = evaluate_golden_set(
        cases,
        retriever=LocalBM25TypologyRetriever(load_typology_documents(args.typologies)),
        annotator=annotator,
        trace_sink=traces,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(golden_summary_as_dict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Spans live beside the summary as JSONL so summary size stays flat as cases grow.
    trace_path = args.output.with_name(f"{args.output.stem}_traces.jsonl")
    trace_path.write_text(
        "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces),
        encoding="utf-8",
    )
    manifest = create_run_manifest(
        output_dir=args.output.parent,
        command="aml-evaluate-golden",
        random_seed=0,
        input_paths={"golden_cases": args.cases, "typology_corpus": args.typologies},
        config_paths=(
            {"prompt_configuration": settings.llm_prompt_config_path}
            if args.use_llm
            else None
        ),
        metadata={
            "use_llm": args.use_llm,
            "case_count": summary.case_count,
            "prompt_versions": summary.prompt_versions,
            "model_names": summary.model_names,
            "hallucination_intercept_rate": summary.hallucination_intercept_rate,
            "no_evidence_refusal_rate": summary.no_evidence_refusal_rate,
            "latency_p50_ms": summary.latency_p50_ms,
            "latency_p95_ms": summary.latency_p95_ms,
        },
        filename=f"{args.output.stem}_run_manifest.json",
    )
    print(
        json.dumps(
            {
                "case_count": summary.case_count,
                "fact_snapshot_match_rate": summary.fact_snapshot_match_rate,
                "hallucination_intercept_rate": summary.hallucination_intercept_rate,
                "no_evidence_refusal_rate": summary.no_evidence_refusal_rate,
                "external_parse_success_rate": summary.external_parse_success_rate,
                "external_fact_validation_pass_rate": (
                    summary.external_fact_validation_pass_rate
                ),
                "latency_p50_ms": summary.latency_p50_ms,
                "latency_p95_ms": summary.latency_p95_ms,
                "run_id": manifest.run_id,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
