"""Optional MLflow adapter over existing aggregate-only run artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def _safe_metric_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.\-/ ]", "_", name.replace("%", "pct"))
    if len(sanitized) <= 240:
        return sanitized
    digest = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()[:12]
    return f"{sanitized[:227]}_{digest}"


def flatten_numeric_metrics(
    payload: dict[str, object], prefix: str = ""
) -> dict[str, float]:
    """Flatten finite numeric leaves while excluding booleans and raw arrays."""
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_numeric_metrics(value, name))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            number = float(value)
            if number == number and abs(number) != float("inf"):
                flattened[_safe_metric_name(name)] = number
    return flattened


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def log_completed_run(
    artifact_dir: Path,
    *,
    experiment_name: str,
    tracking_uri: str,
    pipeline_status: Path | None = None,
    required_pipeline_state: str = "complete",
) -> str:
    """Mirror a completed run into MLflow without logging private input rows."""
    manifest_path = artifact_dir / "run_manifest.json"
    metrics_path = artifact_dir / "metrics.json"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Both run_manifest.json and metrics.json are required.")
    manifest = _load_object(manifest_path)
    metrics = _load_object(metrics_path)
    if pipeline_status is not None:
        if not pipeline_status.is_file():
            raise FileNotFoundError(f"Pipeline status does not exist: {pipeline_status}")
        status = _load_object(pipeline_status)
        if status.get("current_state") != required_pipeline_state:
            raise ValueError(
                "Pipeline is not complete: "
                f"expected {required_pipeline_state!r}, got {status.get('current_state')!r}."
            )
    elif manifest.get("run_purpose") != "full":
        raise ValueError(
            "A full run_purpose or an explicit completed pipeline status is required."
        )
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_manifest.json has no valid run_id.")
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("Install the 'mlops' optional dependency group.") from error

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError("MLflow experiment creation failed.")
    existing = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.source_run_id = '{run_id}' AND attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if not existing.empty:
        existing_run_id = str(existing.iloc[0]["run_id"])
        mlflow.MlflowClient().set_tag(
            existing_run_id, "source_artifact_name", artifact_dir.name
        )
        return existing_run_id
    with mlflow.start_run(run_name=run_id) as active_run:
        mlflow.log_params(
            {
                "source_run_id": run_id,
                "source_revision": manifest.get("source_revision") or "unknown",
                "random_seed": manifest.get("random_seed", "unknown"),
                "schema_version": manifest.get("schema_version", "unknown"),
            }
        )
        mlflow.set_tags(
            {
                "source_run_id": run_id,
                "source_artifact_kind": "aggregate_only",
                "source_artifact_name": artifact_dir.name,
                "source_revision": manifest.get("source_revision") or "unknown",
                "pipeline_state": required_pipeline_state,
                "candidate_selection_scope": "validation_only",
            }
        )
        numeric = flatten_numeric_metrics(metrics)
        if numeric:
            mlflow.log_metrics(numeric)
        mlflow.log_artifact(str(manifest_path), artifact_path="source")
        mlflow.log_artifact(str(metrics_path), artifact_path="source")
        return str(active_run.info.run_id)


def candidate_gate(
    candidate: dict[str, object],
    incumbent: dict[str, object],
    *,
    metric: str = "pr_auc",
    minimum_relative_gain: float = 0.0,
) -> tuple[bool, str]:
    """Gate only on caller-supplied validation metrics; test metrics are not accepted."""
    if "test" in metric.lower():
        raise ValueError("Candidate selection must not use test metrics.")
    candidate_value: Any = candidate.get(metric)
    incumbent_value: Any = incumbent.get(metric)
    if not isinstance(candidate_value, int | float) or isinstance(candidate_value, bool):
        return False, f"candidate_missing_{metric}"
    if not isinstance(incumbent_value, int | float) or isinstance(incumbent_value, bool):
        return False, f"incumbent_missing_{metric}"
    required = float(incumbent_value) * (1.0 + minimum_relative_gain)
    candidate_number = float(candidate_value)
    passed = candidate_number >= required or math.isclose(
        candidate_number,
        required,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )
    return passed, "passed" if passed else "insufficient_validation_gain"
