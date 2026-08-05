#!/usr/bin/env python3
"""Publish aggregate-only evidence for the truncation retry from ignored local runs.

Reads Golden run summaries, which contain model text, and emits counts only. No
annotation, consideration or question text crosses into the published file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-run",
        type=Path,
        required=True,
        help="v7 dev regression against the real provider at the shipped ceiling.",
    )
    parser.add_argument(
        "--forced-run",
        type=Path,
        required=True,
        help="Real run at a lowered ceiling, so truncation is provoked rather than awaited.",
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        required=True,
        help="The v6 run whose single truncation motivated the retry.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> tuple[dict[str, object], str, dict[str, object]]:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    manifest_path = path.parent / f"{path.stem}_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return json.loads(text), hashlib.sha256(text.encode("utf-8")).hexdigest(), manifest


def _aggregate(path: Path, label: str) -> dict[str, object]:
    """Reduce one run summary to counts. Every text field is dropped here."""
    summary, digest, manifest = _load(path)
    cases = [case for case in summary["cases"] if case["external_call_attempted"]]
    return {
        "label": label,
        "run_id": manifest["run_id"],
        "source_revision": manifest["source_revision"],
        "source_summary_sha256": digest,
        "prompt_versions": summary["prompt_versions"],
        "model_names": summary["model_names"],
        "external_case_count": summary["external_case_count"],
        # Absent on runs that predate call-level accounting; a case was one call then.
        "external_call_total": summary.get("external_call_total"),
        "truncation_retry_count": summary.get("truncation_retry_count"),
        "truncation_retry_recovered_count": summary.get("truncation_retry_recovered_count"),
        "external_parse_success_rate": summary["external_parse_success_rate"],
        "external_fact_validation_pass_rate": summary["external_fact_validation_pass_rate"],
        "billable_token_coverage_rate": summary["billable_token_coverage_rate"],
        "billable_prompt_tokens": summary["billable_prompt_tokens"],
        "billable_completion_tokens": summary["billable_completion_tokens"],
        "wasted_prompt_tokens": summary["wasted_prompt_tokens"],
        "wasted_completion_tokens": summary["wasted_completion_tokens"],
        "wasted_token_share": summary["wasted_token_share"],
        "latency_p50_ms": summary["latency_p50_ms"],
        "latency_p95_ms": summary["latency_p95_ms"],
        "error_categories": sorted(
            {
                str(case["external_error_category"])
                for case in cases
                if case["external_error_category"]
            }
        ),
        "retried_cases": [
            {
                "case_id": case["case_id"],
                "attempts": case["external_call_attempts"],
                "wasted_prompt_tokens": case["wasted_prompt_tokens"],
                "wasted_completion_tokens": case["wasted_completion_tokens"],
                "recovered": case["annotation_used"],
            }
            for case in cases
            if (case.get("external_call_attempts") or 1) > 1
        ],
        "completion_token_max": max(
            (case["billable_completion_tokens"] or 0 for case in cases), default=None
        ),
    }


def main() -> None:
    args = parse_args()
    runs = [
        _aggregate(args.baseline_run, "v6_dev_regression34_baseline"),
        _aggregate(args.candidate_run, "v7_dev_regression34_shipped_ceiling"),
        _aggregate(args.forced_run, "diagnostic_forced_truncation_lowered_ceiling"),
    ]
    for run in runs:
        if run["retried_cases"] and not all(
            item["recovered"] for item in run["retried_cases"]
        ):
            # Not an error; recorded so a partial recovery can never be read as total.
            run["all_retries_recovered"] = False

    public = {
        "schema_version": "1.0",
        "evidence_id": "llm-truncation-retry-real-v1",
        "raw_responses_included": False,
        "holdout_cases_used": False,
        "adjudication_independence": "project_internal",
        "runs": runs,
        "findings": [
            "The v7 run at the shipped ceiling fired zero retries: the case that "
            "truncated under v6 completed in 391 tokens this time. Its parse rate of "
            "1.0000 against v6's 0.9630 is therefore NOT attributable to the retry, and "
            "sits well inside the 0.2222 same-configuration spread already recorded for "
            "prompt v3.",
            "Truncation is intermittent run-to-run variance in completion length, not a "
            "fixed property of a case. The ceiling is genuinely tight: the longest of 27 "
            "real completions reached 462 of 500 tokens.",
            "The retry was therefore exercised by lowering the ceiling to 200 rather than "
            "waiting for a natural truncation. All six provoked truncations recovered on "
            "the second attempt, with the fact gate passing on every recovered "
            "annotation.",
            "Recovery is bought, not free: in the provoked run 46.11% of billed tokens "
            "were spent on discarded first attempts.",
        ],
        "limitations": [
            "The provoked run uses a lowered ceiling, so its truncation rate is an "
            "experimental setting and not a field rate.",
            "Every run here is a development-set regression. None is a preregistered "
            "holdout, and none promotes v7.",
            "Review is project-internal, not an external compliance adjudication.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")

    for run in runs:
        print(
            f"  {run['label']:44s} parse={run['external_parse_success_rate']}"
            f" calls={run['external_call_total']}"
            f" retries={run['truncation_retry_count']}"
            f" recovered={run['truncation_retry_recovered_count']}"
        )
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
