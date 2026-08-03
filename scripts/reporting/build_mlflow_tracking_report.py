#!/usr/bin/env python3
"""Build an aggregate-only Chinese report from the local MLflow tracking store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metric(metrics: dict[str, float], scope: str) -> float | None:
    preferred = (
        f"{scope}_metrics.catboost.pr_auc",
        f"{scope}_metrics.pr_auc",
    )
    for name in preferred:
        if name in metrics:
            return float(metrics[name])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="sqlite:///artifacts/mlflow.db")
    parser.add_argument("--experiment", default="aml-evidence-graph-fe-v2")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/mlflow_tracking_report.json")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("artifacts/mlflow_tracking_report.md")
    )
    args = parser.parse_args()
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("Install the 'mlops' optional dependency group.") from error
    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment does not exist: {args.experiment}")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], output_format="list", max_results=1000
    )
    finished: list[dict[str, object]] = []
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        finished.append(
            {
                "mlflow_run_id": run.info.run_id,
                "source_run_id": run.data.tags.get("source_run_id"),
                "source_artifact_name": run.data.tags.get("source_artifact_name"),
                "source_revision": run.data.tags.get("source_revision"),
                "metric_count": len(run.data.metrics),
                "validation_pr_auc": _metric(run.data.metrics, "validation"),
                "test_pr_auc": _metric(run.data.metrics, "test"),
                "candidate_selection_scope": run.data.tags.get(
                    "candidate_selection_scope"
                ),
            }
        )
    finished.sort(key=lambda item: str(item["source_artifact_name"]))
    payload = {
        "schema_version": "1.0",
        "scope": "Local aggregate-only tracking; no transaction rows or model weights.",
        "experiment": args.experiment,
        "run_count": len(runs),
        "finished_count": len(finished),
        "failed_count": sum(run.info.status == "FAILED" for run in runs),
        "finished_runs": finished,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 实验追踪",
        "",
        "> 本报告来自本地 MLflow 聚合指标，不包含交易行、账户标识或模型权重。",
        "",
        f"- Experiment：`{args.experiment}`",
        f"- 总 run：{len(runs)}；FINISHED：{len(finished)}；FAILED：{payload['failed_count']}",
        "- Candidate gate：只允许验证集指标；测试指标不能参与候选选择。",
        "",
        "| 产物 | source run_id | Validation PR-AUC | Test PR-AUC | 指标数 | 状态 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in finished:
        validation = item["validation_pr_auc"]
        test = item["test_pr_auc"]
        lines.append(
            f"| {item['source_artifact_name']} | `{item['source_run_id']}` | "
            f"{validation:.6f} | "
            if isinstance(validation, float)
            else f"| {item['source_artifact_name']} | `{item['source_run_id']}` | — | "
        )
        lines[-1] += (
            f"{test:.6f} | " if isinstance(test, float) else "— | "
        )
        lines[-1] += f"{item['metric_count']} | FINISHED |"
    lines.append("")
    if payload["failed_count"]:
        lines.extend(
            [
                "本地当前保留 FAILED run；接入时曾因 `%` 指标名不符合 MLflow 约束失败，",
                "修复后以只复用 FINISHED run 的幂等规则完成同步。",
            ]
        )
    else:
        lines.append("当前 tracking store 没有 FAILED run；重复同步只复用 FINISHED run。")
    lines.extend(
        [
            "主线 v1 产物缺失，因此当前不标记任何 FE v2 run 为 candidate，也不使用测试集",
            "补做选择。",
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
