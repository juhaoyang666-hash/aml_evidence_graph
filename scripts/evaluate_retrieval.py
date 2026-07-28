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
    evaluate_retriever,
)
from aml_evidence_graph.tracking.run import create_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typologies", type=Path, default=Path("knowledge/typologies"))
    parser.add_argument("--cases", type=Path, default=Path("golden/retrieval_queries_v1.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval/hybrid_v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_evaluation"))
    parser.add_argument("--encoder", choices=("tfidf", "sentence-transformers"), default="tfidf")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {args.output}; pass --overwrite.")
    args.output.mkdir(parents=True, exist_ok=True)
    documents = load_typology_documents(args.typologies)
    raw_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [RetrievalCase.model_validate(item) for item in raw_cases]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Retrieval config must be a YAML object.")
    bm25_config = config.get("bm25", {})
    dense_config = config.get("dense", {})
    reranker_config = config.get("reranker", {})
    if not all(isinstance(item, dict) for item in (bm25_config, dense_config, reranker_config)):
        raise ValueError("Retrieval component configs must be YAML objects.")
    result_limit = int(config.get("result_limit", 3))
    candidate_limit = int(config.get("candidate_limit", 20))
    rrf_constant = int(config.get("rrf_constant", 60))
    corpus = [f"{document.title} {document.body}" for document in documents]
    encoder = (
        TfidfDenseEncoder(corpus)
        if args.encoder == "tfidf"
        else SentenceTransformerEncoder(args.model)
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
    )
    reranker_type = reranker_config.get("type")
    if reranker_type != "lexical_overlap":
        raise ValueError(f"Unsupported reranker type: {reranker_type!r}")
    reranked = HybridTypologyRetriever(
        [bm25, dense],
        reranker=LexicalOverlapReranker(),
        rrf_constant=rrf_constant,
        fetch_limit=candidate_limit,
    )
    summaries = [
        evaluate_retriever(retriever, cases, limit=result_limit)
        for retriever in (bm25, dense, hybrid, reranked)
    ]
    manifest = create_run_manifest(
        output_dir=args.output,
        command="evaluate-retrieval",
        random_seed=20260722,
        input_paths={"typologies": args.typologies, "cases": args.cases},
        config_paths={"retrieval_config": args.config},
        metadata={"encoder": encoder.name, "case_count": len(cases)},
    )
    payload = {
        "schema_version": "1.0",
        "run_id": manifest.run_id,
        "encoder": encoder.name,
        "summaries": [asdict(summary) for summary in summaries],
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
