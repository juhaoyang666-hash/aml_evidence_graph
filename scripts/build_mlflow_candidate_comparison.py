"""Build and log the v1/FE-v2 same-protocol candidate comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from aml_evidence_graph.tracking.candidate_comparison import (
    assert_same_population,
    build_markdown_report,
    create_replayed_evaluation,
    extract_split_metrics,
    load_score_frame,
    log_variant,
    population_summary,
    serialize_manifest,
    validate_expected_population,
)
from aml_evidence_graph.tracking.mlflow_adapter import candidate_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tracking/v1_fe_v2_candidate_comparison.yaml"),
    )
    return parser.parse_args()


def _path(value: str) -> Path:
    return Path(value)


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = _path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol_id": config["protocol_id"],
        "experiment_name": config["experiment_name"],
        "selection_split": config["selection_split"],
        "selection_metric": config["selection_metric"],
        "test_metrics_selection_eligible": False,
        "pairs": [],
    }
    for pair in config["pairs"]:
        frames: dict[str, dict[str, Any]] = {}
        for role in ("incumbent", "candidate"):
            variant = pair[role]
            validation_path = _path(variant["validation_scores"])
            test_path = _path(variant["test_scores"])
            if variant["replay_scores"]:
                validation, test = create_replayed_evaluation(
                    source_artifact_dir=_path(variant["source_artifact_dir"]),
                    output_dir=_path(variant["evaluation_artifact_dir"]),
                    config_path=args.config,
                    validation_path=validation_path,
                    test_path=test_path,
                    score_column=variant["score_column"],
                    variant_id=variant["variant_id"],
                    protocol_id=config["protocol_id"],
                    expected_population=config["expected_population"],
                )
            else:
                validation = load_score_frame(validation_path, variant["score_column"])
                test = load_score_frame(test_path, variant["score_column"])
            validate_expected_population(
                population_summary(validation),
                config["expected_population"]["validation"],
                split="validation",
            )
            validate_expected_population(
                population_summary(test),
                config["expected_population"]["test"],
                split="test",
            )
            frames[role] = {"validation": validation, "test": test}

        validation_sha = assert_same_population(
            frames["incumbent"]["validation"],
            frames["candidate"]["validation"],
            split="validation",
        )
        test_sha = assert_same_population(
            frames["incumbent"]["test"], frames["candidate"]["test"], split="test"
        )
        pair_result: dict[str, Any] = {
            "comparison_group": pair["comparison_group"],
            "interpretation": pair["interpretation"],
            "population": {
                "validation_sha256": validation_sha,
                "test_sha256": test_sha,
                "exact_key_and_label_match": True,
            },
        }
        for role in ("incumbent", "candidate"):
            variant = pair[role]
            artifact_dir = _path(variant["evaluation_artifact_dir"])
            model_name = variant["score_column"] if variant["replay_scores"] else None
            validation_metrics = extract_split_metrics(
                artifact_dir, "validation", model_name=model_name
            )
            if variant.get("test_artifact_dir"):
                test_metrics = extract_split_metrics(_path(variant["test_artifact_dir"]), "test")
            else:
                test_metrics = extract_split_metrics(artifact_dir, "test", model_name=model_name)
            pipeline_status = (
                _path(variant["pipeline_status"]) if variant.get("pipeline_status") else None
            )
            tags = {
                "comparison_protocol_id": config["protocol_id"],
                "comparison_group": pair["comparison_group"],
                "comparison_role": role,
                "variant_id": variant["variant_id"],
                "population_validation_sha256": validation_sha,
                "population_test_sha256": test_sha,
                "selection_split": "validation",
                "selection_metric": "pr_auc",
                "test_metrics_selection_eligible": "false",
            }
            mlflow_run_id = log_variant(
                artifact_dir,
                experiment_name=config["experiment_name"],
                tracking_uri=config["tracking_uri"],
                pipeline_status=pipeline_status,
                required_pipeline_state=variant.get("required_pipeline_state", "complete"),
                tags=tags,
            )
            test_mlflow_run_id = None
            if variant.get("test_artifact_dir"):
                test_tags = dict(tags)
                test_tags["comparison_role"] = f"{role}_test_disclosure"
                test_tags["candidate_selection_scope"] = "test_disclosure_only"
                test_mlflow_run_id = log_variant(
                    _path(variant["test_artifact_dir"]),
                    experiment_name=config["experiment_name"],
                    tracking_uri=config["tracking_uri"],
                    pipeline_status=None,
                    required_pipeline_state="complete",
                    tags=test_tags,
                )
            pair_result[role] = {
                "variant_id": variant["variant_id"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "mlflow_run_id": mlflow_run_id,
                "test_mlflow_run_id": test_mlflow_run_id,
                **serialize_manifest(artifact_dir),
            }
        passed, reason = candidate_gate(
            {"pr_auc": pair_result["candidate"]["validation_pr_auc"]},
            {"pr_auc": pair_result["incumbent"]["validation_pr_auc"]},
            metric="pr_auc",
            minimum_relative_gain=float(config["minimum_relative_gain"]),
        )
        pair_result["validation_absolute_delta"] = (
            pair_result["candidate"]["validation_pr_auc"]
            - pair_result["incumbent"]["validation_pr_auc"]
        )
        pair_result["gate_passed"] = passed
        pair_result["gate_reason"] = reason
        result["pairs"].append(pair_result)

    json_path = output_root / "comparison.json"
    markdown_path = output_root / "comparison.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(build_markdown_report(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
