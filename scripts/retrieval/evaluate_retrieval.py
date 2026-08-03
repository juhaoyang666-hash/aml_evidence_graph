#!/usr/bin/env python3
"""Compare lexical, local dense, hybrid, and reranked Typology retrieval."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from aml_evidence_graph.evidence.typology import load_typology_documents
from aml_evidence_graph.retrieval import (
    BM25TypologyRetriever,
    DenseTypologyRetriever,
    HybridTypologyRetriever,
    LexicalOverlapReranker,
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
        "--cases",
        type=Path,
        nargs="+",
        default=[
            Path("golden/retrieval_queries_v2.json"),
            Path("golden/retrieval_queries_v3_additions.json"),
        ],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/retrieval/hybrid_semantic_v1.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_evaluation"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/检索评估.md"))
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
        help="Exact Hugging Face commit used only by the sentence-transformers encoder.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {args.output}; pass --overwrite.")
    args.output.mkdir(parents=True, exist_ok=True)
    documents = load_typology_documents(args.typologies)
    raw_cases: list[dict[str, object]] = []
    for case_path in args.cases:
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Retrieval cases must be a JSON list: {case_path}")
        raw_cases.extend(payload)
    cases = [RetrievalCase.model_validate(item) for item in raw_cases]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Retrieval case_id values must be unique across all case files.")
    document_ids = {document.typology_id for document in documents}
    unknown_ids = sorted(
        {
            typology_id
            for case in cases
            for typology_id in case.relevant_typology_ids
            if typology_id not in document_ids
        }
    )
    if unknown_ids:
        raise ValueError(f"Retrieval cases reference unknown typologies: {unknown_ids}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Retrieval config must be a YAML object.")
    bm25_config = config.get("bm25", {})
    dense_config = config.get("dense", {})
    reranker_config = config.get("reranker", {})
    hybrid_config = config.get("hybrid", {})
    if not all(
        isinstance(item, dict)
        for item in (bm25_config, dense_config, reranker_config, hybrid_config)
    ):
        raise ValueError("Retrieval component configs must be YAML objects.")
    configured_encoder = dense_config.get("encoder")
    if configured_encoder is not None and configured_encoder != args.encoder:
        raise ValueError(
            f"Config expects encoder {configured_encoder!r}, received {args.encoder!r}."
        )
    result_limit = int(config.get("result_limit", 3))
    candidate_limit = int(config.get("candidate_limit", 20))
    rrf_constant = int(config.get("rrf_constant", 60))
    required_source_prefixes = tuple(hybrid_config.get("required_source_prefixes", []))
    if not all(isinstance(item, str) and item for item in required_source_prefixes):
        raise ValueError("hybrid.required_source_prefixes must contain non-empty strings.")
    corpus = [f"{document.title} {document.body}" for document in documents]
    encoder = (
        TfidfDenseEncoder(corpus)
        if args.encoder == "tfidf"
        else SentenceTransformerEncoder(args.model, revision=args.model_revision)
    )
    bm25 = BM25TypologyRetriever(
        documents,
        minimum_score=float(bm25_config.get("minimum_score", 0.0)),
    )
    dense = DenseTypologyRetriever(
        documents,
        encoder,
        minimum_score=float(dense_config.get("minimum_score", 1e-12)),
    )
    hybrid = HybridTypologyRetriever(
        [bm25, dense],
        rrf_constant=rrf_constant,
        fetch_limit=candidate_limit,
        required_source_prefixes=required_source_prefixes,
    )
    reranker_type = reranker_config.get("type")
    if reranker_type != "lexical_overlap":
        raise ValueError(f"Unsupported reranker type: {reranker_type!r}")
    reranked = HybridTypologyRetriever(
        [bm25, dense],
        reranker=LexicalOverlapReranker(),
        rrf_constant=rrf_constant,
        fetch_limit=candidate_limit,
        required_source_prefixes=required_source_prefixes,
    )
    evaluations: list[dict[str, object]] = []
    all_tags = sorted({tag for case in cases for tag in case.tags})
    for retriever in (bm25, dense, hybrid, reranked):
        case_results = evaluate_retriever_cases(retriever, cases, limit=result_limit)
        summary = summarize_retrieval_results(retriever.name, case_results)
        tag_summaries = {
            tag: asdict(
                summarize_retrieval_results(
                    retriever.name,
                    [result for result in case_results if tag in result.tags],
                )
            )
            for tag in all_tags
        }
        evaluations.append(
            {
                "summary": asdict(summary),
                "tag_summaries": tag_summaries,
                "bad_cases": [
                    asdict(result) for result in case_results if not result.passed
                ],
                "case_results": [asdict(result) for result in case_results],
            }
        )
    manifest = create_run_manifest(
        output_dir=args.output,
        command="evaluate-retrieval",
        random_seed=20260722,
        input_paths={
            "typologies": args.typologies,
            **{f"cases_{index}": path for index, path in enumerate(args.cases, start=1)},
        },
        config_paths={"retrieval_config": args.config},
        metadata={
            "encoder": encoder.name,
            "case_count": len(cases),
            "tags": all_tags,
            "document_count": len(documents),
        },
    )
    payload = {
        "schema_version": "1.0",
        "run_id": manifest.run_id,
        "encoder": encoder.name,
        "evaluations": evaluations,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Typology 检索评估",
        "",
        "> 检索结果只作为调查线索，不参与风险评分或案件结论。",
        "",
        f"- Golden：{', '.join(f'`{path.as_posix()}`' for path in args.cases)}"
        f"（合计 {len(cases)} 条）",
        f"- Typology 文档：{len(documents)} 篇",
        f"- Encoder：`{encoder.name}`",
        f"- Config：`{args.config.as_posix()}`",
        f"- run_id：`{manifest.run_id}`",
        "",
        "| 检索器 | Recall@1 | Recall@3 | MRR | nDCG@3 | 无答案误召回率 | Bad Case |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for evaluation in evaluations:
        summary = evaluation["summary"]
        bad_cases = evaluation["bad_cases"]
        lines.append(
            f"| {summary['retriever']} | {summary['recall_at_1']:.3f} | "
            f"{summary['recall_at_3']:.3f} | {summary['mean_reciprocal_rank']:.3f} | "
            f"{summary['ndcg_at_3']:.3f} | "
            f"{summary['no_answer_false_positive_rate']:.3f} | {len(bad_cases)} |"
        )
    lines.extend(
        [
            "",
            "## 关键切片",
            "",
            "| 检索器 | 有来源 Recall@3 | 有来源 MRR | 中文 Recall@3 | "
            "Hard-negative Recall@3 | 无答案误召回率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for evaluation in evaluations:
        summary = evaluation["summary"]
        slices = evaluation["tag_summaries"]
        lines.append(
            f"| {summary['retriever']} | "
            f"{slices['source-grounded']['recall_at_3']:.3f} | "
            f"{slices['source-grounded']['mean_reciprocal_rank']:.3f} | "
            f"{slices['zh']['recall_at_3']:.3f} | "
            f"{slices['hard-negative']['recall_at_3']:.3f} | "
            f"{slices['no-answer']['no_answer_false_positive_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 语料来源",
            "",
            "| Typology | 版本 | 标题 | 来源 |",
            "|---|---|---|---|",
        ]
    )
    for document in documents:
        lines.append(
            f"| `{document.typology_id}` | `{document.version}` | {document.title} | "
            f"{document.source} |"
        )
    lines.extend(
        [
            "",
            "## 边界与晋升规则",
            "",
            "- `direct/paraphrase/zh/hard-negative/no-answer` 等分组明细保存在聚合产物中。",
            "- 只有 hybrid/rerank 在重复评测中稳定优于 BM25 时才允许切换默认检索器。",
            "- `hybrid_semantic_v1.yaml` 的 0.35 拒答阈值在 v3 增量集之前冻结；本轮未按新"
            " hard negative 调参。",
            "- TF-IDF 对照必须显式使用 `--encoder tfidf --config "
            "configs/retrieval/hybrid_v1.yaml`；脚本拒绝 encoder/config 静默混用。",
            "- 当前 Golden 是项目作者构建的公开合成裁定集，不代表合规专家生产验收。",
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
