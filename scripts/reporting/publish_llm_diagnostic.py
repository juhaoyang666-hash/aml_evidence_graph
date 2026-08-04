#!/usr/bin/env python3
"""Publish aggregate-only LLM diagnostic evidence from an ignored local run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--dev-regression", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostic_text = args.diagnostic.read_text(encoding="utf-8").replace("\r\n", "\n")
    diagnostic = json.loads(diagnostic_text)
    diagnostic_manifest_path = (
        args.diagnostic.parent / f"{args.diagnostic.stem}_run_manifest.json"
    )
    diagnostic_manifest = json.loads(
        diagnostic_manifest_path.read_text(encoding="utf-8")
    )
    if diagnostic.get("raw_responses_publishable") is not False:
        raise ValueError("Diagnostic must explicitly mark raw responses non-publishable.")
    protocol_bytes = (
        args.protocol.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    )
    public = {
        "schema_version": "1.0",
        "purpose": diagnostic["purpose"],
        "generated_at": diagnostic["generated_at"],
        "diagnostic_not_model_evaluation": True,
        "holdout_cases_used": False,
        "raw_responses_included": False,
        "protocol_path": args.protocol.as_posix(),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_summary_sha256": hashlib.sha256(
            diagnostic_text.encode("utf-8")
        ).hexdigest(),
        "diagnostic_run_id": diagnostic_manifest["run_id"],
        "source_revision": diagnostic_manifest["source_revision"],
        "summary_by_variant": diagnostic["summary_by_variant"],
        "interpretation": (
            "The token-limit control isolates truncation risk on synthetic diagnostic "
            "inputs; it does not identify every historical Holdout failure or establish "
            "candidate quality."
        ),
    }
    if args.dev_regression is not None:
        regression_text = args.dev_regression.read_text(encoding="utf-8").replace(
            "\r\n", "\n"
        )
        regression = json.loads(regression_text)
        regression_manifest_path = (
            args.dev_regression.parent
            / f"{args.dev_regression.stem}_run_manifest.json"
        )
        regression_manifest = json.loads(
            regression_manifest_path.read_text(encoding="utf-8")
        )
        public["v5_development_regression"] = {
            "run_id": regression_manifest["run_id"],
            "source_summary_sha256": hashlib.sha256(
                regression_text.encode("utf-8")
            ).hexdigest(),
            "case_count": regression["case_count"],
            "external_parse_success_rate": regression["external_parse_success_rate"],
            "external_fact_validation_pass_rate": regression[
                "external_fact_validation_pass_rate"
            ],
            "llm_annotation_rate": regression["llm_annotation_rate"],
            "latency_p50_ms": regression["latency_p50_ms"],
            "latency_p95_ms": regression["latency_p95_ms"],
            "prompt_versions": regression["prompt_versions"],
            "development_set_reused": True,
            "candidate_promoted": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
