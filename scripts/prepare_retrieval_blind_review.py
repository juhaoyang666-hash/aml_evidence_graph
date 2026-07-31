#!/usr/bin/env python3
"""Build a prediction-free packet for independent retrieval adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aml_evidence_graph.evidence.typology import load_typology_documents
from aml_evidence_graph.retrieval.blind_review import (
    assert_no_prior_overlap,
    load_blind_queries,
    sha256_directory,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/retrieval/blind_review_v1.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/retrieval_blind_v1_packet"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite blind-review packet: {args.output}")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") not in {
        "frozen_awaiting_independent_adjudication",
        "frozen_awaiting_project_adjudication",
    }:
        raise ValueError("Blind-review protocol is not frozen")
    paths = protocol["paths"]
    hashes = protocol["sha256"]
    checks = {
        "retrieval_config": sha256_file(Path(paths["retrieval_config"])),
        "calibration_cases": sha256_file(Path(paths["calibration_cases"])),
        "retrospective_cases": sha256_file(Path(paths["retrospective_cases"])),
        "typology_corpus": sha256_directory(Path(paths["typology_corpus"])),
        "gate_model": sha256_file(Path(paths["gate_model"])),
    }
    mismatches = {
        key: {"expected": hashes[key], "actual": value}
        for key, value in checks.items()
        if hashes.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen protocol integrity failure: {mismatches}")

    queries = load_blind_queries(args.queries)
    minimum_cases = int(protocol["evaluation"]["minimum_cases"])
    if len(queries) < minimum_cases:
        raise ValueError(f"Blind set requires at least {minimum_cases} queries")
    prior_paths = [Path(item) for item in protocol["paths"]["prior_case_files"]]
    assert_no_prior_overlap(queries, prior_paths)
    documents = load_typology_documents(Path(paths["typology_corpus"]))

    args.output.mkdir(parents=True)
    packet = {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol),
        "instructions": [
            "Judge each query using only the frozen typology catalog.",
            "Do not inspect model predictions, retrieval artifacts, v2/v3 labels, or metrics.",
            "Use no_answer when none of the catalog entries is supported.",
            "Use exclude only for malformed or genuinely ambiguous queries and explain why.",
        ],
        "queries": [item.model_dump() for item in queries],
        "typology_catalog": [
            {
                "typology_id": item.typology_id,
                "version": item.version,
                "title": item.title,
                "body": item.body,
                "source": item.source,
            }
            for item in documents
        ],
    }
    packet_path = args.output / "review_packet.json"
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    template = {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "packet_sha256": sha256_file(packet_path),
        "adjudicator_id": "REPLACE_WITH_PSEUDONYMOUS_ID",
        "independent_from_system_development": protocol["adjudication"][
            "independent_from_system_development"
        ],
        "blind_to_model_outputs": None,
        "completed_at": "REPLACE_WITH_ISO_8601_TIMESTAMP",
        "judgments": [
            {
                "case_id": item.case_id,
                "decision": "answerable|no_answer|exclude",
                "relevant_typology_ids": [],
                "confidence": "high|medium|low",
                "rationale": "REQUIRED",
            }
            for item in queries
        ],
    }
    (args.output / "adjudication_template.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"packet": str(packet_path), "cases": len(queries)}))


if __name__ == "__main__":
    main()
