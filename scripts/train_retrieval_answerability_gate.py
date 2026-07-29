#!/usr/bin/env python3
"""Train an OOF-calibrated answerability gate without changing risk scoring."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typologies", type=Path, default=Path("knowledge/typologies"))
    parser.add_argument(
        "--calibration", type=Path, default=Path("golden/retrieval_queries_v2.json")
    )
    parser.add_argument(
        "--retrospective-test",
        type=Path,
        default=Path("golden/retrieval_queries_v3_additions.json"),
    )
    parser.add_argument("--dense-threshold", type=float, default=0.35)
    parser.add_argument("--target-no-answer-fpr", type=float, default=0.20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/retrieval_answerability_gate_v1"),
    )
    return parser.parse_args()


def _load_cases(path: Path) -> list[RetrievalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [RetrievalCase.model_validate(item) for item in payload]


def _retrieval_features(
    cases: list[RetrievalCase],
    encoder: SentenceTransformerEncoder,
    dense: DenseTypologyRetriever,
    bm25: BM25TypologyRetriever,
) -> np.ndarray:
    embeddings = encoder.encode([case.query for case in cases])
    signals: list[list[float]] = []
    for case in cases:
        dense_hits = dense.search(case.query, limit=3)
        bm25_hits = bm25.search(case.query, limit=3)
        dense_scores = [hit.score for hit in dense_hits]
        bm25_scores = [hit.score for hit in bm25_hits]
        non_ascii = sum(ord(char) > 127 for char in case.query)
        signals.append(
            [
                dense_scores[0] if dense_scores else 0.0,
                dense_scores[0] - dense_scores[1] if len(dense_scores) > 1 else 0.0,
                bm25_scores[0] if bm25_scores else 0.0,
                bm25_scores[0] - bm25_scores[1] if len(bm25_scores) > 1 else 0.0,
                float(len(case.query)),
                non_ascii / max(len(case.query), 1),
            ]
        )
    return np.column_stack([embeddings, np.asarray(signals, dtype=np.float32)])


def _model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2_000,
            random_state=20260722,
        ),
    )


class _ProbabilityGatedRetriever:
    def __init__(
        self,
        base: HybridTypologyRetriever,
        probabilities: dict[str, float],
        threshold: float,
    ) -> None:
        self.base = base
        self.probabilities = probabilities
        self.threshold = threshold
        self.name = "hybrid-rrf+answerability-gate"

    def search(self, query: str, *, limit: int = 3) -> list[object]:
        if self.probabilities[query] < self.threshold:
            return []
        return self.base.search(query, limit=limit)


def _select_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    target_fpr: float,
) -> tuple[float, list[dict[str, float]]]:
    thresholds = sorted({0.0, 1.0, *(float(value) for value in probabilities)})
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        predicted_answerable = probabilities >= threshold
        answerable_tpr = float(predicted_answerable[labels == 1].mean())
        no_answer_fpr = float(predicted_answerable[labels == 0].mean())
        rows.append(
            {
                "threshold": threshold,
                "answerable_acceptance_rate": answerable_tpr,
                "no_answer_false_positive_rate": no_answer_fpr,
            }
        )
    eligible = [row for row in rows if row["no_answer_false_positive_rate"] <= target_fpr]
    if not eligible:
        raise RuntimeError("No answerability threshold satisfies the no-answer FPR target.")
    selected = max(
        eligible,
        key=lambda row: (
            row["answerable_acceptance_rate"],
            -row["no_answer_false_positive_rate"],
            row["threshold"],
        ),
    )
    return float(selected["threshold"]), rows


def _summary(
    cases: list[RetrievalCase],
    probabilities: np.ndarray,
    threshold: float,
    base: HybridTypologyRetriever,
) -> dict[str, object]:
    probability_map = {
        case.query: float(probability)
        for case, probability in zip(cases, probabilities, strict=True)
    }
    retriever = _ProbabilityGatedRetriever(base, probability_map, threshold)
    return asdict(
        summarize_retrieval_results(
            retriever.name,
            evaluate_retriever_cases(retriever, cases, limit=3),
        )
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite gate output: {args.output}")
    args.output.mkdir(parents=True)
    calibration = _load_cases(args.calibration)
    retrospective = _load_cases(args.retrospective_test)
    if {case.case_id for case in calibration} & {case.case_id for case in retrospective}:
        raise ValueError("Calibration and retrospective case IDs must be disjoint.")

    documents = load_typology_documents(args.typologies)
    encoder = SentenceTransformerEncoder(MODEL, revision=REVISION)
    dense = DenseTypologyRetriever(documents, encoder, minimum_score=0.0)
    bm25 = BM25TypologyRetriever(documents, minimum_score=0.0)
    hybrid_dense = DenseTypologyRetriever(
        documents, encoder, minimum_score=args.dense_threshold
    )
    hybrid = HybridTypologyRetriever(
        [bm25, hybrid_dense],
        rrf_constant=60,
        fetch_limit=20,
        required_source_prefixes=("dense:",),
    )

    x_calibration = _retrieval_features(calibration, encoder, dense, bm25)
    y_calibration = np.asarray(
        [not case.expect_no_answer for case in calibration], dtype=np.int64
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260722)
    oof_probabilities = np.zeros(len(calibration), dtype=np.float64)
    for train_indices, holdout_indices in splitter.split(x_calibration, y_calibration):
        fold_model = _model()
        fold_model.fit(x_calibration[train_indices], y_calibration[train_indices])
        oof_probabilities[holdout_indices] = fold_model.predict_proba(
            x_calibration[holdout_indices]
        )[:, 1]
    threshold, threshold_grid = _select_threshold(
        oof_probabilities, y_calibration, args.target_no_answer_fpr
    )

    final_model = _model()
    final_model.fit(x_calibration, y_calibration)
    x_retrospective = _retrieval_features(retrospective, encoder, dense, bm25)
    retrospective_probabilities = final_model.predict_proba(x_retrospective)[:, 1]
    payload = {
        "schema_version": "1.0",
        "model": f"{MODEL}@{REVISION}",
        "feature_contract": (
            "normalized embedding + dense top/margin + BM25 top/margin + length/language"
        ),
        "protocol": {
            "calibration": str(args.calibration),
            "calibration_method": "5-fold stratified OOF",
            "target_no_answer_false_positive_rate": args.target_no_answer_fpr,
            "dense_threshold": args.dense_threshold,
            "retrospective_test": str(args.retrospective_test),
            "retrospective_is_independent_frozen_test": False,
            "note": (
                "v3 motivated this gate and is diagnostic only; a new independent set is required."
            ),
        },
        "selected_answerability_threshold": threshold,
        "threshold_grid": threshold_grid,
        "calibration_oof": _summary(
            calibration, oof_probabilities, threshold, hybrid
        ),
        "retrospective_v3": _summary(
            retrospective, retrospective_probabilities, threshold, hybrid
        ),
    }
    joblib.dump(final_model, args.output / "answerability_gate.joblib")
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
