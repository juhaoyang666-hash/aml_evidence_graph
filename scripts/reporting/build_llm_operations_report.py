#!/usr/bin/env python3
"""Build the external-LLM operations and cost report from frozen run summaries.

Reads only local run summaries; performs no external calls and copies no model
text, so the output is safe to publish alongside the other public evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from aml_evidence_graph.evaluation.llm_operations import (
    build_operations_report,
    load_run_summary,
    summarize_run_operations,
)

# Frozen runs in promotion order. Labels stay stable so the report diffs cleanly.
DEFAULT_RUNS: tuple[tuple[str, str], ...] = (
    ("prompt_v1_golden34_frozen", "artifacts/llm_ecnu_max_20260804/golden_34_summary.json"),
    (
        "prompt_v2_regression4",
        "artifacts/llm_ecnu_max_20260804/prompt_v2_regression4_summary.json",
    ),
    (
        "prompt_v3_regression4",
        "artifacts/llm_ecnu_max_20260804/prompt_v3_regression4_summary.json",
    ),
    (
        "prompt_v3_golden34_regression_a",
        "artifacts/llm_ecnu_max_20260804/prompt_v3_golden34_regression_summary.json",
    ),
    (
        "prompt_v3_golden34_regression_b_published",
        "artifacts/llm_ecnu_max_20260804/prompt_v3_golden34_final_summary.json",
    ),
    ("prompt_v3_holdout_v1", "artifacts/llm_holdout_v1/blind_run_summary.json"),
    ("prompt_v4_holdout_v2", "artifacts/llm_holdout_v2/blind_run_summary.json"),
    ("prompt_v5_dev_regression6", "artifacts/llm_diagnostics/v5_dev_regression.json"),
    ("prompt_v6_dev_regression34", "artifacts/llm_diagnostics/v6_full_dev_regression.json"),
    ("prompt_v6_holdout_v3_promoted", "artifacts/llm_holdout_v3/blind_run_summary.json"),
    # First run executed after the billable-basis fix, so usage covers failed calls too.
    (
        "prompt_v6_dev_regression34_complete_usage",
        "artifacts/llm_operations/v6_dev_regression_complete_usage.json",
    ),
    # v7 candidate at the shipped ceiling. Fired zero retries: see the report findings.
    (
        "prompt_v7_dev_regression34_real",
        "artifacts/llm_operations/v7_dev_regression34.json",
    ),
    # Ceiling lowered to 200 so truncation is provoked instead of awaited. The retry
    # rate here is an experimental setting, not a field rate.
    (
        "diagnostic_forced_truncation_real",
        "artifacts/llm_diagnostics/forced_truncation_retry_real.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/llm_operations/llm_operations_report.json"),
    )
    parser.add_argument(
        "--report-id",
        default="ecnu-max-llm-operations-v1",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip runs whose summary file is absent instead of failing.",
    )
    parser.add_argument(
        "--input-cost-per-million-tokens-usd",
        type=float,
        default=None,
        help="Contract input price. Applied post hoc to recorded tokens; omit to keep cost null.",
    )
    parser.add_argument(
        "--output-cost-per-million-tokens-usd",
        type=float,
        default=None,
        help="Contract output price. Must be supplied together with the input price.",
    )
    arguments = parser.parse_args()
    priced = (
        arguments.input_cost_per_million_tokens_usd is not None,
        arguments.output_cost_per_million_tokens_usd is not None,
    )
    if any(priced) and not all(priced):
        parser.error("Input and output prices must be supplied together.")
    return arguments


def main() -> None:
    args = parse_args()
    runs = []
    missing: list[str] = []
    for label, relative_path in DEFAULT_RUNS:
        path = Path(relative_path)
        if not path.is_file():
            missing.append(relative_path)
            continue
        summary, run_id = load_run_summary(path)
        runs.append(summarize_run_operations(summary, run_label=label, run_id=run_id))
    if missing and not args.allow_missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing frozen run summaries: {joined}")
    if not runs:
        raise SystemExit("No frozen run summaries were found.")

    report = build_operations_report(
        runs,
        report_id=args.report_id,
        generated_at=datetime.now(UTC).isoformat(),
        input_cost_per_million_tokens_usd=args.input_cost_per_million_tokens_usd,
        output_cost_per_million_tokens_usd=args.output_cost_per_million_tokens_usd,
    )
    if missing:
        report["skipped_missing_runs"] = missing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"runs: {len(runs)}  output: {args.output}")
    print(
        "external calls: "
        f"{report['external_calls_total']}  usage observed: "
        f"{report['usage_observed_calls_total']}  unrecoverable: "
        f"{report['unrecoverable_usage_calls_total']}"
    )
    for run in runs:
        rate = run.parse_success_rate
        parse = "n/a" if rate is None else f"{rate:.4f}"
        print(
            f"  {run.run_label:42s} parse={parse}"
            f" avail_fail={run.availability_failures} fmt_fail={run.format_failures}"
            f" cost={run.cost_status}"
        )
    for item in report["post_hoc_pricing"] or []:
        floor = ">=" if item["billable_cost_is_lower_bound"] else "=="
        wasted = item["wasted_cost_usd_from_recorded_tokens"]
        wasted_text = (
            f"${wasted:.6f}"
            if wasted is not None
            else f"unmeasured({item['unmeasured_calls']} calls)"
        )
        print(
            f"  priced {item['run_label']:42s}"
            f" billable{floor}${item['billable_cost_usd_from_recorded_tokens']:.6f}"
            f" wasted={wasted_text}"
        )
    for item in report["repeated_configuration_variance"]:
        print(
            f"  variance {item['prompt_version']} n={item['run_count']}: "
            f"parse {round(item['parse_success_rate_min'], 4)}"
            f"–{round(item['parse_success_rate_max'], 4)}"
            f" (spread {round(item['parse_success_rate_spread'], 4)})"
        )


if __name__ == "__main__":
    main()
