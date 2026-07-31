"""Auditable same-population candidate comparisons for completed model runs."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.tracking.mlflow_adapter import log_completed_run
from aml_evidence_graph.tracking.run import create_run_manifest, fingerprint_path

KEY_COLUMNS = ["transaction_id", "event_ts", "is_laundering"]


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_score_frame(path: Path, score_column: str) -> pl.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    required = [*KEY_COLUMNS, score_column]
    schema = pl.scan_parquet(path).collect_schema()
    missing = [column for column in required if column not in schema]
    if missing:
        raise ValueError(f"Missing score columns in {path}: {missing}")
    frame = pl.read_parquet(path, columns=required)
    if frame["transaction_id"].null_count() or frame["transaction_id"].n_unique() != len(frame):
        raise ValueError(f"transaction_id must be non-null and unique: {path}")
    labels = set(frame["is_laundering"].unique().to_list())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError(f"Both binary label classes are required: {path}")
    invalid_scores = frame.select(
        (~pl.col(score_column).is_finite())
        | pl.col(score_column).is_null()
        | (pl.col(score_column) < 0)
        | (pl.col(score_column) > 1)
    ).to_series()
    if invalid_scores.any():
        raise ValueError(f"Scores must be finite probabilities in [0, 1]: {path}")
    return frame


def population_summary(frame: pl.DataFrame) -> dict[str, Any]:
    ordered = frame.select(KEY_COLUMNS).sort("transaction_id")
    row_hashes = ordered.hash_rows(seed=0).to_numpy().tobytes()
    minimum = ordered["event_ts"].min()
    maximum = ordered["event_ts"].max()
    return {
        "sample_count": len(ordered),
        "positive_count": int(ordered["is_laundering"].sum()),
        "start_date": minimum.date().isoformat(),
        "end_date": maximum.date().isoformat(),
        "sha256": hashlib.sha256(row_hashes).hexdigest(),
        "fingerprint_method": "polars-hash-rows-seed-0-sorted-transaction-id-v1",
    }


def validate_expected_population(
    actual: dict[str, Any], expected: dict[str, Any], *, split: str
) -> None:
    for field in ("sample_count", "positive_count", "start_date", "end_date"):
        expected_value = expected[field]
        if field.endswith("date"):
            expected_value = date.fromisoformat(str(expected_value)).isoformat()
        if actual[field] != expected_value:
            raise ValueError(
                f"{split} population mismatch for {field}: "
                f"expected {expected_value!r}, got {actual[field]!r}"
            )


def assert_same_population(
    incumbent: pl.DataFrame, candidate: pl.DataFrame, *, split: str
) -> str:
    left = incumbent.select(KEY_COLUMNS).sort("transaction_id")
    right = candidate.select(KEY_COLUMNS).sort("transaction_id")
    if not left.equals(right, null_equal=True):
        raise ValueError(f"{split} candidate and incumbent populations are not identical")
    return population_summary(left)["sha256"]


def extract_split_metrics(
    artifact_dir: Path, split: str, *, model_name: str | None = None
) -> dict[str, Any]:
    metrics = load_json_object(artifact_dir / "metrics.json")
    value: Any = metrics.get(f"{split}_metrics")
    if model_name is not None and isinstance(value, dict):
        value = value.get(model_name)
    if not isinstance(value, dict):
        raise ValueError(f"No {split} metrics found in {artifact_dir}")
    return value


def create_replayed_evaluation(
    *,
    source_artifact_dir: Path,
    output_dir: Path,
    config_path: Path,
    validation_path: Path,
    test_path: Path,
    score_column: str,
    variant_id: str,
    protocol_id: str,
    expected_population: dict[str, dict[str, Any]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    validation = load_score_frame(validation_path, score_column)
    test = load_score_frame(test_path, score_column)
    validation_population = population_summary(validation)
    test_population = population_summary(test)
    validate_expected_population(
        validation_population, expected_population["validation"], split="validation"
    )
    validate_expected_population(test_population, expected_population["test"], split="test")

    if output_dir.exists() and any(output_dir.iterdir()):
        existing = load_json_object(output_dir / "metrics.json")
        existing_manifest = load_json_object(output_dir / "run_manifest.json")
        replay = existing.get("replay", {})
        if not isinstance(replay, dict) or replay.get("protocol_id") != protocol_id:
            raise FileExistsError(f"Refusing to overwrite incompatible replay: {output_dir}")
        current_inputs = {
            "validation_scores": fingerprint_path(validation_path),
            "test_scores": fingerprint_path(test_path),
            "source_manifest": fingerprint_path(source_artifact_dir / "run_manifest.json"),
        }
        if existing_manifest.get("inputs") != current_inputs:
            raise ValueError(f"Replay inputs changed since evaluation: {output_dir}")
        if existing.get("validation_population") != validation_population:
            raise ValueError(f"Validation population changed since replay: {output_dir}")
        if existing.get("test_population") != test_population:
            raise ValueError(f"Test population changed since replay: {output_dir}")
        return validation, test

    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = source_artifact_dir / "run_manifest.json"
    source_manifest = load_json_object(source_manifest_path)
    manifest = create_run_manifest(
        output_dir=output_dir,
        command=f"candidate-comparison replay {variant_id}",
        random_seed=0,
        input_paths={
            "validation_scores": validation_path,
            "test_scores": test_path,
            "source_manifest": source_manifest_path,
        },
        config_paths={"comparison": config_path},
        run_purpose="full",
        metadata={
            "variant_id": variant_id,
            "protocol_id": protocol_id,
            "source_run_id": source_manifest.get("run_id"),
            "source_revision": source_manifest.get("source_revision"),
            "source_manifest_fingerprint": fingerprint_path(source_manifest_path),
            "private_rows_logged": False,
        },
    )
    payload = {
        "run_id": manifest.run_id,
        "validation_metrics": {
            score_column: evaluate_binary_risk_scores(
                validation["is_laundering"].to_numpy(),
                validation[score_column].to_numpy(),
            )
        },
        "test_metrics": {
            score_column: evaluate_binary_risk_scores(
                test["is_laundering"].to_numpy(), test[score_column].to_numpy()
            )
        },
        "validation_population": validation_population,
        "test_population": test_population,
        "replay": {
            "protocol_id": protocol_id,
            "variant_id": variant_id,
            "score_column": score_column,
            "source_run_id": source_manifest.get("run_id"),
            "aggregate_only": True,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return validation, test


def tag_comparison_run(run_id: str, tags: dict[str, str], tracking_uri: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    for key, value in tags.items():
        client.set_tag(run_id, key, value)


def log_variant(
    artifact_dir: Path,
    *,
    experiment_name: str,
    tracking_uri: str,
    pipeline_status: Path | None,
    required_pipeline_state: str,
    tags: dict[str, str],
) -> str:
    run_id = log_completed_run(
        artifact_dir,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        pipeline_status=pipeline_status,
        required_pipeline_state=required_pipeline_state,
    )
    tag_comparison_run(run_id, tags, tracking_uri)
    return run_id


def serialize_manifest(artifact_dir: Path) -> dict[str, Any]:
    manifest = load_json_object(artifact_dir / "run_manifest.json")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "source_run_id": manifest.get("run_id"),
        "source_revision": metadata.get("source_revision", manifest.get("source_revision")),
        "evaluation_revision": manifest.get("source_revision"),
        "upstream_source_run_id": metadata.get("source_run_id"),
        "run_purpose": manifest.get("run_purpose"),
    }


def build_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# v1 / FE v2 同协议 MLflow Candidate 对照",
        "",
        f"- 协议：`{result['protocol_id']}`",
        f"- MLflow experiment：`{result['experiment_name']}`",
        "- 选型口径：仅验证集 `PR-AUC`；测试集仅披露，不参与 candidate gate。",
        "- 对照边界：Windows 本机、相同时间切分与逐行人群；不声称与 Linux 主线字节级复现。",
        "",
        "| 对照组 | v1/基准验证 PR-AUC | FE v2/候选验证 PR-AUC | "
        "绝对变化 | Gate | 测试 PR-AUC（基准 → 候选） |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for pair in result["pairs"]:
        lines.append(
            f"| {pair['comparison_group']} | {pair['incumbent']['validation_pr_auc']:.6f} "
            f"| {pair['candidate']['validation_pr_auc']:.6f} | "
            f"{pair['validation_absolute_delta']:+.6f} "
            f"| {'PASS' if pair['gate_passed'] else 'FAIL'} | "
            f"{pair['incumbent']['test_pr_auc']:.6f} → {pair['candidate']['test_pr_auc']:.6f} |"
        )
    lines.extend(["", "## 审计结论", ""])
    for pair in result["pairs"]:
        lines.extend(
            [
                f"### {pair['comparison_group']}",
                "",
                f"{pair['interpretation']}",
                f"人群指纹一致：验证 `{pair['population']['validation_sha256']}`，"
                f"测试 `{pair['population']['test_sha256']}`。",
                f"MLflow run：基准 `{pair['incumbent']['mlflow_run_id']}`，"
                f"候选 `{pair['candidate']['mlflow_run_id']}`。",
                "",
            ]
        )
    lines.extend(
        [
            "## 结果解读",
            "",
            "CatBoost 受益于 FE v2；GAT FE v2 未超过 v1 replay。在固定 CatBoost FE v2 "
            "和融合协议时，GAT v1 组件的融合结果仍更强。因此当前不应用 GAT FE v2 "
            "替换 GAT v1；这不否定 FE v2 表格特征的增益。",
            "",
        ]
    )
    return "\n".join(lines)
