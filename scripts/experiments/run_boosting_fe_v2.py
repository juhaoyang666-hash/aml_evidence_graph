"""Compare LightGBM and XGBoost against CatBoost FE v2 under one frozen protocol."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aml_evidence_graph.data.contract import CANONICAL  # noqa: E402
from aml_evidence_graph.data.splits import TimeSplit  # noqa: E402
from aml_evidence_graph.evaluation.metrics import (  # noqa: E402
    compare_alert_volume_at_fixed_recall,
    evaluate_binary_risk_scores,
)
from aml_evidence_graph.models.boosting import (  # noqa: E402
    CategoryVocabulary,
    fit_category_vocabulary,
    prepare_native_boosting_frame,
)
from aml_evidence_graph.models.tabular import FeatureSpec, infer_feature_spec  # noqa: E402
from aml_evidence_graph.tracking.run import create_run_manifest  # noqa: E402
from aml_evidence_graph.training.table_baseline import (  # noqa: E402
    deterministic_negative_downsample,
    load_feature_split,
    rule_baseline_scores,
)


@dataclass(frozen=True)
class CandidateResult:
    family: str
    name: str
    parameters: dict[str, Any]
    validation_metrics: dict[str, Any]
    best_iteration: int
    fit_seconds: float
    model_path: str


def _candidate_definitions(random_seed: int, scale_pos_weight: float) -> list[dict[str, Any]]:
    common = {
        "learning_rate": 0.05,
        "n_estimators": 1_200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_seed,
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": 12,
    }
    return [
        {
            "family": "lightgbm",
            "name": "lightgbm_leaves31",
            "params": {
                **common,
                "objective": "binary",
                "metric": "None",
                "num_leaves": 31,
                "min_child_samples": 80,
                "max_depth": -1,
                "verbosity": -1,
                "deterministic": True,
                "force_col_wise": True,
            },
        },
        {
            "family": "lightgbm",
            "name": "lightgbm_leaves63",
            "params": {
                **common,
                "objective": "binary",
                "metric": "None",
                "num_leaves": 63,
                "min_child_samples": 100,
                "max_depth": -1,
                "verbosity": -1,
                "deterministic": True,
                "force_col_wise": True,
            },
        },
        {
            "family": "xgboost",
            "name": "xgboost_depth6",
            "params": {
                **common,
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "max_depth": 6,
                "min_child_weight": 5,
                "max_bin": 256,
                "tree_method": "hist",
                "device": "cuda",
                "enable_categorical": True,
            },
        },
        {
            "family": "xgboost",
            "name": "xgboost_depth8",
            "params": {
                **common,
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "max_depth": 8,
                "min_child_weight": 10,
                "max_bin": 256,
                "tree_method": "hist",
                "device": "cuda",
                "enable_categorical": True,
            },
        },
    ]


def _json_safe_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in parameters.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _fit_candidate(
    candidate: dict[str, Any],
    train_x: Any,
    train_y: np.ndarray,
    validation_x: Any,
    validation_y: np.ndarray,
    model_dir: Path,
) -> tuple[Any, CandidateResult, np.ndarray]:
    family = str(candidate["family"])
    name = str(candidate["name"])
    params = dict(candidate["params"])
    started = time.perf_counter()
    if family == "lightgbm":
        model = lgb.LGBMClassifier(**params)
        model.fit(
            train_x,
            train_y,
            eval_set=[(validation_x, validation_y)],
            eval_metric="average_precision",
            categorical_feature="auto",
            callbacks=[
                lgb.early_stopping(100, first_metric_only=True, verbose=False),
                lgb.log_evaluation(50),
            ],
        )
        best_iteration = int(model.best_iteration_ or params["n_estimators"])
    elif family == "xgboost":
        try:
            model = xgb.XGBClassifier(**params, early_stopping_rounds=100)
            model.fit(train_x, train_y, eval_set=[(validation_x, validation_y)], verbose=False)
        except xgb.core.XGBoostError as error:
            if "cuda" not in str(error).lower():
                raise
            params["device"] = "cpu"
            model = xgb.XGBClassifier(**params, early_stopping_rounds=100)
            model.fit(train_x, train_y, eval_set=[(validation_x, validation_y)], verbose=False)
        best_iteration = int(getattr(model, "best_iteration", params["n_estimators"] - 1)) + 1
    else:
        raise ValueError(f"Unsupported candidate family: {family}")
    fit_seconds = time.perf_counter() - started
    scores = model.predict_proba(validation_x)[:, 1].astype(np.float64, copy=False)
    metrics = evaluate_binary_risk_scores(validation_y, scores)
    model_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, model_dir / "model.joblib", compress=3)
    native_path = model_dir / ("model.txt" if family == "lightgbm" else "model.json")
    if family == "lightgbm":
        model.booster_.save_model(native_path, num_iteration=best_iteration)
    else:
        model.save_model(native_path)
    result = CandidateResult(
        family=family,
        name=name,
        parameters=_json_safe_parameters(params),
        validation_metrics=metrics,
        best_iteration=best_iteration,
        fit_seconds=fit_seconds,
        model_path=str(model_dir),
    )
    return model, result, scores


def _predict_in_batches(
    model: Any,
    frame: pl.DataFrame,
    spec: FeatureSpec,
    vocabulary: CategoryVocabulary,
    *,
    batch_rows: int,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for offset in range(0, frame.height, batch_rows):
        batch = frame.slice(offset, batch_rows)
        batch_x = prepare_native_boosting_frame(batch, spec, vocabulary)
        pieces.append(model.predict_proba(batch_x)[:, 1].astype(np.float64, copy=False))
        del batch_x
    return np.concatenate(pieces)


def _write_scores(path: Path, frame: pl.DataFrame, scores: dict[str, np.ndarray]) -> None:
    output = frame.select(
        CANONICAL.transaction_id,
        CANONICAL.event_ts,
        CANONICAL.is_laundering,
    )
    for name, values in scores.items():
        output = output.with_columns(pl.Series(name, values))
    output.write_parquet(path)


def run_experiment(
    feature_root: Path,
    output_dir: Path,
    *,
    maximum_training_negative_rows: int = 500_000,
    random_seed: int = 20260722,
    batch_rows: int = 200_000,
    validation_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "models").mkdir()
    (output_dir / "scores").mkdir()

    full_training = load_feature_split(feature_root, TimeSplit.TRAIN)
    training_rows_before_sampling = full_training.height
    sampled_training = deterministic_negative_downsample(
        full_training,
        maximum_negative_rows=maximum_training_negative_rows,
    )
    validation = load_feature_split(feature_root, TimeSplit.VALIDATION)
    spec = infer_feature_spec(sampled_training, excluded_prefixes=("graph_",))
    vocabulary = fit_category_vocabulary(sampled_training, spec)
    train_x = prepare_native_boosting_frame(sampled_training, spec, vocabulary)
    validation_x = prepare_native_boosting_frame(validation, spec, vocabulary)
    train_y = sampled_training[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    validation_y = validation[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    positives = int(train_y.sum())
    scale_pos_weight = float((len(train_y) - positives) / positives)

    candidate_results: list[CandidateResult] = []
    validation_scores: dict[str, np.ndarray] = {}
    for candidate in _candidate_definitions(random_seed, scale_pos_weight):
        model, result, scores = _fit_candidate(
            candidate,
            train_x,
            train_y,
            validation_x,
            validation_y,
            output_dir / "models" / str(candidate["name"]),
        )
        candidate_results.append(result)
        validation_scores[result.name] = scores
        print(
            json.dumps(
                {
                    "candidate": result.name,
                    "validation_pr_auc": result.validation_metrics["pr_auc"],
                    "best_iteration": result.best_iteration,
                    "fit_seconds": result.fit_seconds,
                }
            ),
            flush=True,
        )
        del model
        gc.collect()

    winners = {
        family: max(
            (result for result in candidate_results if result.family == family),
            key=lambda result: float(result.validation_metrics["pr_auc"]),
        )
        for family in ("lightgbm", "xgboost")
    }
    _write_scores(
        output_dir / "scores" / "validation_scores.parquet",
        validation,
        validation_scores,
    )

    del train_x, validation_x, full_training
    gc.collect()
    test_metrics: dict[str, dict[str, Any]] = {}
    alert_reduction: dict[str, dict[str, Any]] = {}
    test_rows: int | None = None
    if not validation_only:
        # Test is loaded only after both family winners are fixed on validation.
        test = load_feature_split(feature_root, TimeSplit.TEST)
        test_rows = test.height
        test_y = test[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
        test_scores: dict[str, np.ndarray] = {}
        rule_scores = rule_baseline_scores(test)
        for family, winner in winners.items():
            model = joblib.load(Path(winner.model_path) / "model.joblib")
            scores = _predict_in_batches(
                model,
                test,
                spec,
                vocabulary,
                batch_rows=batch_rows,
            )
            test_scores[family] = scores
            test_metrics[family] = evaluate_binary_risk_scores(test_y, scores)
            if rule_scores is not None:
                alert_reduction[family] = compare_alert_volume_at_fixed_recall(
                    test_y,
                    scores,
                    rule_scores,
                )
            del model
            gc.collect()
        _write_scores(output_dir / "scores" / "test_scores.parquet", test, test_scores)

    summary = {
        "protocol": {
            "feature_root": str(feature_root),
            "training_sampling": "stable_hash_negative_downsample",
            "maximum_training_negative_rows": maximum_training_negative_rows,
            "feature_exclusion_prefixes": ["graph_"],
            "selection_split": "validation_only",
            "test_access": (
                "not_accessed" if validation_only else "once_after_family_winner_selection"
            ),
            "random_seed": random_seed,
        },
        "rows": {
            "training_before_sampling": training_rows_before_sampling,
            "training_after_sampling": sampled_training.height,
            "validation": validation.height,
            "test": test_rows,
            "training_positives": positives,
            "scale_pos_weight": scale_pos_weight,
        },
        "features": {
            "numeric": list(spec.numeric_columns),
            "categorical": list(spec.categorical_columns),
        },
        "package_versions": {
            "lightgbm": version("lightgbm"),
            "xgboost": version("xgboost"),
        },
        "candidates": [asdict(result) for result in candidate_results],
        "selected": {family: winner.name for family, winner in winners.items()},
        "test_metrics": test_metrics,
        "alert_reduction_vs_rules": alert_reduction,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = create_run_manifest(
        output_dir=output_dir,
        command="python scripts/experiments/run_boosting_fe_v2.py",
        random_seed=random_seed,
        input_paths={"feature_root": feature_root},
        metadata={
            "selected": summary["selected"],
            "selection_split": "validation_only",
            "test_access": summary["protocol"]["test_access"],
        },
        run_purpose="full",
    )
    summary["run_id"] = manifest.run_id
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/pit_features_fe_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/boosting_fe_v2"))
    parser.add_argument("--maximum-training-negative-rows", type=int, default=500_000)
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument("--batch-rows", type=int, default=200_000)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        args.feature_root,
        args.output_dir,
        maximum_training_negative_rows=args.maximum_training_negative_rows,
        random_seed=args.random_seed,
        batch_rows=args.batch_rows,
        validation_only=args.validation_only,
        overwrite=args.overwrite,
    )
    print(json.dumps({"run_id": summary["run_id"], "selected": summary["selected"]}))


if __name__ == "__main__":
    main()
