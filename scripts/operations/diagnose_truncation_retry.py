#!/usr/bin/env python3
"""Measure the truncation retry against a simulated provider, with no external calls.

The real defect appeared once in 27 calls, which is too rare to measure a fix against.
This harness replaces the provider with a deterministic stub whose completion length is
chosen per case, so the truncation rate is set rather than sampled. It therefore proves
the mechanism, not the field rate: what a real run would do depends on how often the
provider actually overruns the ceiling, which only a priced run can tell you.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from aml_evidence_graph.evidence.package import RiskEvidencePackage
from aml_evidence_graph.evidence.typology import LocalBM25TypologyRetriever, TypologyDocument
from aml_evidence_graph.investigation.golden import GoldenCase, evaluate_golden_set
from aml_evidence_graph.investigation.llm import (
    ECNUAnnotationClient,
    PromptConfiguration,
    load_prompt_configuration,
)
from aml_evidence_graph.tracking.run import create_run_manifest

# Priced so the artifact carries a cost column. These are the DeepSeek V4 Flash public
# cache-miss rates used elsewhere as a labelled sensitivity, never as a paid price.
INPUT_PRICE = 0.14
OUTPUT_PRICE = 0.28

VALID_ANNOTATION = {
    "evidence_references": ["model_probabilities.catboost"],
    "analytical_considerations": [
        "Withheld model values cannot indicate a direction on their own.",
        "Corroborating context from authorized source records remains outstanding.",
    ],
    "recommended_questions": [
        "Which authorized source records could corroborate the retained categories?",
        "What corroborating context is available for the withheld feature categories?",
    ],
}


class SimulatedProvider:
    """Return a truncated body whenever the completion exceeds the requested ceiling.

    This is the whole point of the harness: truncation becomes a function of the token
    ceiling instead of provider luck, so raising the ceiling on retry has an observable,
    repeatable effect.
    """

    def __init__(self, lengths: list[int]) -> None:
        self._lengths = lengths
        self.calls: list[tuple[int, int]] = []

    def response_for(self, case_index: int, max_tokens: int) -> dict[str, object]:
        length = self._lengths[case_index]
        self.calls.append((max_tokens, length))
        if length > max_tokens:
            content = '{"evidence_references": [], "analytical_considerations": ['
            finish_reason = "length"
            completion_tokens = max_tokens
        else:
            content = json.dumps(VALID_ANNOTATION, ensure_ascii=False)
            finish_reason = "stop"
            completion_tokens = length
        return {
            "usage": {
                "prompt_tokens": 700,
                "completion_tokens": completion_tokens,
                "total_tokens": 700 + completion_tokens,
            },
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        }


class _StubClient:
    """Drive the real client's retry logic against the simulated provider.

    The production client is used unmodified; only its transport is replaced, so the
    retry path exercised here is the one that would run against a real provider.
    """

    def __init__(self, provider: SimulatedProvider, configuration: PromptConfiguration) -> None:
        self._provider = provider
        self._case_index = -1
        self._client = ECNUAnnotationClient(
            api_key="simulated",
            http_client=httpx.Client(transport=httpx.MockTransport(self._handle)),
            input_cost_per_million_tokens_usd=INPUT_PRICE,
            output_cost_per_million_tokens_usd=OUTPUT_PRICE,
            prompt_configuration=configuration,
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json=self._provider.response_for(self._case_index, payload["max_tokens"]),
        )

    def annotate(self, evidence, references):  # noqa: ANN001, ANN201 - protocol shape
        # Each Golden case is one annotate() call, so advancing here keeps the scripted
        # completion length aligned with the case even across a retry.
        self._case_index += 1
        return self._client.annotate(evidence, references)


def _retriever() -> LocalBM25TypologyRetriever:
    return LocalBM25TypologyRetriever(
        [
            TypologyDocument(
                typology_id="T-SIM",
                version="1",
                title="Simulated typology",
                source="diagnostic",
                body="Transaction risk investigation.",
            )
        ]
    )


def _cases(count: int) -> list[GoldenCase]:
    return [
        GoldenCase(
            case_id=f"sim-{index:02d}",
            case_category="typology",
            evidence=RiskEvidencePackage(
                alert_id=f"sim-alert-{index:02d}",
                generated_at=datetime(2026, 8, 5, tzinfo=UTC),
                transaction_id=f"sim-txn-{index:02d}",
                event_timestamp=datetime(2023, 7, 1, tzinfo=UTC),
                model_probabilities={"catboost": 0.8},
            ),
        )
        for index in range(count)
    ]


def _completion_lengths(count: int, overruns: int, *, base_ceiling: int) -> list[int]:
    """Fit under the base ceiling except for a fixed number of deliberate overruns.

    Overruns land between the base and retry ceilings, which is the case the retry is
    designed for. A body longer than both ceilings would truncate twice, and that path
    is covered by the regression tests rather than measured here.
    """
    fits = max(1, base_ceiling - 160)
    overrun = base_ceiling + 120
    return [overrun if index < overruns else fits for index in range(count)]


def _run_arm(
    label: str,
    configuration: PromptConfiguration,
    *,
    case_count: int,
    overruns: int,
) -> dict[str, object]:
    provider = SimulatedProvider(
        _completion_lengths(case_count, overruns, base_ceiling=configuration.max_tokens)
    )
    summary = evaluate_golden_set(
        _cases(case_count),
        retriever=_retriever(),
        annotator=_StubClient(provider, configuration),
    )
    return {
        "arm": label,
        "prompt_version": configuration.version,
        "max_tokens": configuration.max_tokens,
        "truncation_retry_max_tokens": configuration.truncation_retry_max_tokens,
        "case_count": summary.case_count,
        "simulated_overruns": overruns,
        "external_case_count": summary.external_case_count,
        "external_call_total": summary.external_call_total,
        "truncation_retry_count": summary.truncation_retry_count,
        "truncation_retry_recovered_count": summary.truncation_retry_recovered_count,
        "external_parse_success_rate": summary.external_parse_success_rate,
        "external_fact_validation_pass_rate": summary.external_fact_validation_pass_rate,
        "billable_token_coverage_rate": summary.billable_token_coverage_rate,
        "billable_prompt_tokens": summary.billable_prompt_tokens,
        "billable_completion_tokens": summary.billable_completion_tokens,
        "billable_estimated_cost_usd": summary.billable_estimated_cost_usd,
        "wasted_prompt_tokens": summary.wasted_prompt_tokens,
        "wasted_completion_tokens": summary.wasted_completion_tokens,
        "wasted_estimated_cost_usd": summary.wasted_estimated_cost_usd,
        "wasted_token_share": summary.wasted_token_share,
        "error_categories": sorted(
            {
                case.external_error_category
                for case in summary.cases
                if case.external_error_category
            }
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-count", type=int, default=20)
    parser.add_argument(
        "--overruns",
        type=int,
        default=4,
        help="Cases whose completion exceeds the base ceiling. Set, not sampled.",
    )
    parser.add_argument(
        "--baseline-prompt",
        type=Path,
        default=Path("configs/prompts/ecnu-risk-evidence-v6.yaml"),
    )
    parser.add_argument(
        "--candidate-prompt",
        type=Path,
        default=Path("configs/prompts/ecnu-risk-evidence-v7.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/llm_diagnostics/truncation_retry_matrix.json"),
    )
    arguments = parser.parse_args()
    if not 0 <= arguments.overruns <= arguments.case_count:
        parser.error("--overruns must be between 0 and --case-count.")
    return arguments


def main() -> None:
    args = parse_args()
    baseline = load_prompt_configuration(args.baseline_prompt)
    candidate = load_prompt_configuration(args.candidate_prompt)
    if candidate.truncation_retry_max_tokens is None:
        raise SystemExit("Candidate prompt has no truncation_retry_max_tokens; nothing to test.")
    if candidate.system_instructions != baseline.system_instructions:
        raise SystemExit(
            "Baseline and candidate instructions differ, so any delta would confound "
            "the retry with a prompt change."
        )

    arms = [
        _run_arm("baseline_no_retry", baseline, case_count=args.case_count, overruns=args.overruns),
        _run_arm("candidate_retry", candidate, case_count=args.case_count, overruns=args.overruns),
    ]
    report = {
        "schema_version": "1.0",
        "diagnostic_id": "llm-truncation-retry-matrix-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": "simulated_deterministic_stub",
        "external_calls_made": 0,
        "scope": (
            "Mechanism check on a simulated provider. Truncation frequency is set by the "
            "harness, so parse rates here are not field rates and must not be reported as "
            "provider performance."
        ),
        "reference_prices_usd_per_million_tokens": {
            "input": INPUT_PRICE,
            "output": OUTPUT_PRICE,
            "basis": (
                "DeepSeek V4 Flash public cache-miss rate, applied as a labelled "
                "sensitivity. This project paid nothing for its calls."
            ),
        },
        "arms": arms,
        "limitations": [
            "A simulated provider cannot show how often a real completion overruns the "
            "ceiling, only what the chain does when one does.",
            "Bodies exceeding both ceilings truncate twice; that path is covered by "
            "tests/investigation/test_truncation_retry.py, not by this matrix.",
            "Content quality is unchanged by construction, because the stub returns the "
            "same annotation whenever it fits.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    create_run_manifest(
        output_dir=args.output.parent,
        command="diagnose-truncation-retry",
        random_seed=0,
        input_paths={
            "baseline_prompt": args.baseline_prompt,
            "candidate_prompt": args.candidate_prompt,
        },
        metadata={"case_count": args.case_count, "overruns": args.overruns},
        filename=f"{args.output.stem}_run_manifest.json",
    )

    for arm in arms:
        print(
            f"{arm['arm']:20s} parse={arm['external_parse_success_rate']:.4f}"
            f" calls={arm['external_call_total']}"
            f" retries={arm['truncation_retry_count']}"
            f" recovered={arm['truncation_retry_recovered_count']}"
            f" billable=${arm['billable_estimated_cost_usd']:.6f}"
            f" wasted=${arm['wasted_estimated_cost_usd']:.6f}"
            f" ({arm['wasted_token_share']:.4f} of tokens)"
        )
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
