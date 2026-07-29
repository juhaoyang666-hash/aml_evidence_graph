"""Calibrate semantic abstention on v2 and evaluate once on frozen v3 cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aml_evidence_graph.evidence.typology import load_typology_documents
from aml_evidence_graph.retrieval import (
    BM25TypologyRetriever,
    DenseTypologyRetriever,
    HybridTypologyRetriever,
    RetrievalCase,
    SentenceTransformerEncoder,
    TfidfDenseEncoder,
    evaluate_retriever_cases,
    summarize_retrieval_results,
)
from aml_evidence_graph.tracking.run import create_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typologies", type=Path, default=Path("knowledge/typologies"))
    parser.add_argument(
        "--calibration-cases",
        type=Path,
        default=Path("golden/retrieval_queries_v2.json"),
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=Path("golden/retrieval_queries_v3_additions.json"),
    )
    parser.add_argument("--target-no-answer-fpr", type=float, default=0.20)
    parser.add_argument("--threshold-start", type=float, default=0.30)
    parser.add_argument("--threshold-stop", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument(
        "--encoder",
        choices=("tfidf", "sentence-transformers"),
        default="sentence-transformers",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument(
        "--model-revision",
        default="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_abstention"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/检索拒答校准.md"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_cases(path: Path) -> list[RetrievalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Retrieval cases must be a JSON list: {path}")
    return [RetrievalCase.model_validate(item) for item in payload]


def main() -> None:
    args = parse_args()
    if not 0 <= args.target_no_answer_fpr <= 1:
        raise ValueError("target-no-answer-fpr must be in [0, 1].")
    if args.threshold_step <= 0 or args.threshold_stop < args.threshold_start:
        raise ValueError("Invalid threshold grid.")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {args.output}; pass --overwrite.")
    args.output.mkdir(parents=True, exist_ok=True)

    documents = load_typology_documents(args.typologies)
    calibration_cases = _load_cases(args.calibration_cases)
    test_cases = _load_cases(args.test_cases)
    calibration_ids = {case.case_id for case in calibration_cases}
    test_ids = {case.case_id for case in test_cases}
    if calibration_ids.intersection(test_ids):
        raise ValueError("Calibration and test case IDs must be disjoint.")
    corpus = [f"{document.title} {document.body}" for document in documents]
    encoder = (
        TfidfDenseEncoder(corpus)
        if args.encoder == "tfidf"
        else SentenceTransformerEncoder(args.model, revision=args.model_revision)
    )
    bm25 = BM25TypologyRetriever(documents, minimum_score=0.0)
    dense = DenseTypologyRetriever(documents, encoder, minimum_score=args.threshold_start)
    hybrid = HybridTypologyRetriever(
        [bm25, dense],
        rrf_constant=60,
        fetch_limit=20,
        required_source_prefixes=("dense:",),
    )

    threshold_count = int(round((args.threshold_stop - args.threshold_start) / args.threshold_step))
    thresholds = [
        round(args.threshold_start + index * args.threshold_step, 10)
        for index in range(threshold_count + 1)
    ]
    calibration_rows: list[dict[str, float]] = []
    for threshold in thresholds:
        dense.minimum_score = threshold
        results = evaluate_retriever_cases(hybrid, calibration_cases, limit=3)
        summary = summarize_retrieval_results(hybrid.name, results)
        calibration_rows.append(
            {
                "threshold": threshold,
                "recall_at_3": summary.recall_at_3,
                "mrr": summary.mean_reciprocal_rank,
                "no_answer_false_positive_rate": summary.no_answer_false_positive_rate,
            }
        )
    eligible = [
        row
        for row in calibration_rows
        if row["no_answer_false_positive_rate"] <= args.target_no_answer_fpr
    ]
    if not eligible:
        raise RuntimeError("No threshold satisfies the calibration no-answer FPR target.")
    selected = max(
        eligible,
        key=lambda row: (
            row["recall_at_3"],
            row["mrr"],
            -row["no_answer_false_positive_rate"],
            row["threshold"],
        ),
    )
    dense.minimum_score = selected["threshold"]
    test_results = evaluate_retriever_cases(hybrid, test_cases, limit=3)
    test_summary = summarize_retrieval_results(hybrid.name, test_results)
    manifest = create_run_manifest(
        output_dir=args.output,
        command="calibrate-retrieval-abstention",
        random_seed=20260722,
        input_paths={
            "typologies": args.typologies,
            "calibration_cases": args.calibration_cases,
            "test_cases": args.test_cases,
        },
        metadata={
            "encoder": encoder.name,
            "selection_metric": "max_recall_at_3_subject_to_no_answer_fpr",
        },
    )
    payload = {
        "schema_version": "1.0",
        "run_id": manifest.run_id,
        "encoder": encoder.name,
        "protocol": {
            "calibration_cases": str(args.calibration_cases),
            "test_cases": str(args.test_cases),
            "disjoint_case_ids": True,
            "target_no_answer_false_positive_rate": args.target_no_answer_fpr,
        },
        "calibration_grid": calibration_rows,
        "selected": selected,
        "frozen_test": asdict(test_summary),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Typology 检索拒答校准",
        "",
        "> v2 只用于选择拒答阈值；v3 case ID 完全隔离，并在阈值冻结后只评估一次。",
        "",
        f"- Encoder：`{encoder.name}`",
        f"- Calibration：`{args.calibration_cases}`（{len(calibration_cases)} 条）",
        f"- Frozen test：`{args.test_cases}`（{len(test_cases)} 条）",
        f"- 目标无答案误召回率：`≤ {args.target_no_answer_fpr:.3f}`",
        f"- 选定 dense threshold：`{selected['threshold']:.3f}`",
        "",
        "| 数据集 | Recall@3 | MRR | 无答案误召回率 |",
        "|---|---:|---:|---:|",
        f"| v2 calibration | {selected['recall_at_3']:.3f} | {selected['mrr']:.3f} | "
        f"{selected['no_answer_false_positive_rate']:.3f} |",
        f"| v3 frozen test | {test_summary.recall_at_3:.3f} | "
        f"{test_summary.mean_reciprocal_rank:.3f} | "
        f"{test_summary.no_answer_false_positive_rate:.3f} |",
        "",
        "该阈值只控制 Typology 检索是否拒答，不改变交易风险分数、融合或案件结论。",
        "当前 Golden 为项目构建的公开合成裁定集，不是独立合规专家生产验收。",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
