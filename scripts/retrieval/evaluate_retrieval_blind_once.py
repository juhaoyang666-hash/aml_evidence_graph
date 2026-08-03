#!/usr/bin/env python3
"""Evaluate the frozen two-stage retriever once on completed blind judgments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import yaml

from aml_evidence_graph.evidence.typology import load_typology_documents
from aml_evidence_graph.retrieval import (
    BM25TypologyRetriever,
    DenseTypologyRetriever,
    HybridTypologyRetriever,
    RetrievalCase,
    SentenceTransformerEncoder,
    evaluate_retriever_cases,
    summarize_retrieval_results,
)
from aml_evidence_graph.retrieval.answerability import (
    ANSWERABILITY_FEATURE_CONTRACT,
    ProbabilityGatedRetriever,
    build_answerability_features,
)
from aml_evidence_graph.retrieval.blind_review import (
    BlindJudgment,
    BlindQuery,
    assert_no_prior_overlap,
    sha256_directory,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/retrieval/blind_review_v1.yaml"),
    )
    return parser.parse_args()


def _verify_frozen_files(protocol: dict[str, object]) -> None:
    paths = protocol["paths"]
    hashes = protocol["sha256"]
    actual = {
        "retrieval_config": sha256_file(Path(paths["retrieval_config"])),
        "calibration_cases": sha256_file(Path(paths["calibration_cases"])),
        "retrospective_cases": sha256_file(Path(paths["retrospective_cases"])),
        "typology_corpus": sha256_directory(Path(paths["typology_corpus"])),
        "gate_model": sha256_file(Path(paths["gate_model"])),
    }
    mismatch = {key: value for key, value in actual.items() if hashes.get(key) != value}
    if mismatch:
        raise ValueError(f"Frozen protocol integrity failure: {mismatch}")


def main() -> None:
    args = parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") not in {
        "frozen_awaiting_independent_adjudication",
        "frozen_awaiting_project_adjudication",
    }:
        raise ValueError("Blind-review protocol is not frozen")
    output = Path(protocol["evaluation"]["canonical_output"])
    if output.exists():
        raise FileExistsError(
            f"Canonical blind evaluation already exists; one-shot rerun refused: {output}"
        )
    _verify_frozen_files(protocol)

    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    adjudication = json.loads(args.adjudication.read_text(encoding="utf-8"))
    if packet.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Packet protocol_id does not match frozen protocol")
    if packet.get("protocol_sha256") != sha256_file(args.protocol):
        raise ValueError("Packet does not refer to this exact frozen protocol")
    if adjudication.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Adjudication protocol_id does not match frozen protocol")
    if adjudication.get("packet_sha256") != sha256_file(args.packet):
        raise ValueError("Adjudication does not refer to this exact review packet")
    expected_independence = protocol["adjudication"][
        "independent_from_system_development"
    ]
    if adjudication.get("independent_from_system_development") is not expected_independence:
        raise ValueError("Adjudicator independence declaration differs from protocol")
    if adjudication.get("blind_to_model_outputs") is not True:
        raise ValueError("Model-blind attestation is required")
    adjudicator_id = str(adjudication.get("adjudicator_id", ""))
    if not adjudicator_id or adjudicator_id.startswith("REPLACE_"):
        raise ValueError("A pseudonymous adjudicator_id is required")
    try:
        datetime.fromisoformat(str(adjudication["completed_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise ValueError("A valid ISO-8601 completed_at value is required") from exc

    queries = [BlindQuery.model_validate(item) for item in packet["queries"]]
    query_ids = [item.case_id for item in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Packet query case_id values must be unique")
    assert_no_prior_overlap(
        queries, [Path(item) for item in protocol["paths"]["prior_case_files"]]
    )
    judgments = [BlindJudgment.model_validate(item) for item in adjudication["judgments"]]
    judgment_map = {item.case_id: item for item in judgments}
    if len(judgment_map) != len(judgments):
        raise ValueError("Adjudication case_id values must be unique")
    if set(judgment_map) != {item.case_id for item in queries}:
        raise ValueError("Adjudication must cover every packet case exactly once")

    included = [item for item in queries if judgment_map[item.case_id].decision != "exclude"]
    cases = [
        RetrievalCase(
            case_id=item.case_id,
            query=item.query,
            relevant_typology_ids=judgment_map[item.case_id].relevant_typology_ids,
            expect_no_answer=judgment_map[item.case_id].decision == "no_answer",
            tags=item.tags,
        )
        for item in included
    ]
    answerable = sum(not item.expect_no_answer for item in cases)
    no_answer = sum(item.expect_no_answer for item in cases)
    evaluation_config = protocol["evaluation"]
    if len(cases) < int(evaluation_config["minimum_cases"]):
        raise ValueError("Too many excluded cases: minimum evaluation size not met")
    if answerable < int(evaluation_config["minimum_answerable_cases"]):
        raise ValueError("Minimum answerable-case count not met")
    if no_answer < int(evaluation_config["minimum_no_answer_cases"]):
        raise ValueError("Minimum no-answer-case count not met")

    documents = load_typology_documents(Path(protocol["paths"]["typology_corpus"]))
    known_typology_ids = {item.typology_id for item in documents}
    unknown_typology_ids = sorted(
        {
            typology_id
            for judgment in judgments
            for typology_id in judgment.relevant_typology_ids
            if typology_id not in known_typology_ids
        }
    )
    if unknown_typology_ids:
        raise ValueError(f"Judgments reference unknown typologies: {unknown_typology_ids}")
    system = protocol["frozen_system"]
    if system["feature_contract"] != ANSWERABILITY_FEATURE_CONTRACT:
        raise ValueError("Runtime answerability feature contract differs from frozen protocol")
    encoder = SentenceTransformerEncoder(system["encoder_model"], revision=system["revision"])
    dense_raw = DenseTypologyRetriever(documents, encoder, minimum_score=0.0)
    dense_gated = DenseTypologyRetriever(
        documents, encoder, minimum_score=float(system["dense_threshold"])
    )
    bm25 = BM25TypologyRetriever(documents, minimum_score=0.0)
    hybrid = HybridTypologyRetriever(
        [bm25, dense_gated],
        rrf_constant=int(system["rrf_constant"]),
        fetch_limit=int(system["candidate_limit"]),
        required_source_prefixes=("dense:",),
    )
    features = build_answerability_features(cases, encoder, dense_raw, bm25)
    model = joblib.load(Path(protocol["paths"]["gate_model"]))
    probabilities = np.asarray(model.predict_proba(features))[:, 1]
    probability_map = {
        case.query: float(value) for case, value in zip(cases, probabilities, strict=True)
    }
    gated = ProbabilityGatedRetriever(
        hybrid, probability_map, float(system["answerability_threshold"])
    )
    limit = int(evaluation_config["result_limit"])
    baseline_results = evaluate_retriever_cases(hybrid, cases, limit=limit)
    gated_results = evaluate_retriever_cases(gated, cases, limit=limit)
    baseline = asdict(summarize_retrieval_results(hybrid.name, baseline_results))
    gated_summary = asdict(summarize_retrieval_results(gated.name, gated_results))
    criteria = evaluation_config["promotion_criteria"]
    checks = {
        "no_answer_fpr_target": gated_summary["no_answer_false_positive_rate"]
        <= float(criteria["maximum_no_answer_fpr"]),
        "no_answer_not_worse_than_baseline": gated_summary[
            "no_answer_false_positive_rate"
        ]
        <= baseline["no_answer_false_positive_rate"],
        "recall_drop_within_limit": gated_summary["recall_at_3"]
        >= baseline["recall_at_3"] - float(criteria["maximum_recall_at_3_drop"]),
        "mrr_drop_within_limit": gated_summary["mean_reciprocal_rank"]
        >= baseline["mean_reciprocal_rank"] - float(criteria["maximum_mrr_drop"]),
    }
    payload = {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "one_shot": True,
        "input_sha256": {
            "protocol": sha256_file(args.protocol),
            "packet": sha256_file(args.packet),
            "adjudication": sha256_file(args.adjudication),
        },
        "feature_contract": ANSWERABILITY_FEATURE_CONTRACT,
        "cases": {"included": len(cases), "answerable": answerable, "no_answer": no_answer},
        "baseline": baseline,
        "two_stage_gate": gated_summary,
        "promotion_checks": checks,
        "promote": all(checks.values()),
        "case_results": {
            "baseline": [asdict(item) for item in baseline_results],
            "two_stage_gate": [asdict(item) for item in gated_results],
        },
    }
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "promote": payload["promote"]}))


if __name__ == "__main__":
    main()
