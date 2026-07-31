#!/usr/bin/env python3
"""Run the pre-registered GraphSAGE FE-v2/novelty validation matrix serially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/experiments/graphsage_fe_v2_novelty_matrix.json"),
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("artifacts/logs/graphsage_fe_v2_novelty_matrix_status.json"),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _run(command: list[str], *, root: Path, dry_run: bool) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=root, check=True)


def _validate_metrics(path: Path, *, expected_seed: int) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("selection_scope") != "train_validation_only":
        raise ValueError(f"Not a validation-only artifact: {path}")
    if payload.get("test_split_read") is not False:
        raise ValueError(f"Artifact may have read test: {path}")
    configuration = payload.get("configuration", {})
    if configuration.get("architecture") != "graphsage":
        raise ValueError(f"Expected GraphSAGE architecture: {path}")
    if configuration.get("random_seed") != expected_seed:
        raise ValueError(f"Unexpected random seed: {path}")
    return payload


def _job_commands(
    *,
    root: Path,
    python: Path,
    spec: dict[str, Any],
    job: dict[str, Any],
    output: Path,
) -> tuple[list[str], list[str]]:
    controlled = spec["controlled_configuration"]
    candidate = [
        str(python),
        "scripts/run_gat_validation_candidate.py",
        "--features",
        str(spec["features"]),
        "--output",
        str(output.relative_to(root)),
        "--model-config",
        str(spec["model_config"]),
        "--device",
        str(controlled["device"]),
        "--batch-size",
        str(controlled["batch_size"]),
        "--num-neighbors",
        *[str(value) for value in controlled["num_neighbors"]],
        "--history-window-days",
        str(controlled["history_window_days"]),
        "--random-seed",
        str(job["seed"]),
    ]
    for family in job["exclude_cold_start_families"]:
        candidate.extend(["--exclude-cold-start-family", str(family)])
    slices = [
        str(python),
        "scripts/evaluate_gat_risk_slices.py",
        "--features",
        str(spec["features"]),
        "--scores",
        str((output / "scores/graphsage_validation_scores.parquet").relative_to(root)),
        "--output",
        str((output / "validation_risk_slices.json").relative_to(root)),
        "--split",
        "validation",
    ]
    return candidate, slices


def _metric_row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["validation_metrics"]
    budget = metrics["alert_budgets"]["0.1000%"]
    return {
        "name": name,
        "architecture": payload["configuration"]["architecture"],
        "seed": payload["configuration"]["random_seed"],
        "pr_auc": metrics["pr_auc"],
        "precision_at_0_1_percent": budget["precision_at_k"],
        "recall_at_0_1_percent": budget["recall_at_k"],
        "test_split_read": payload["test_split_read"],
    }


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    spec_path = (root / args.spec).resolve()
    status_path = (root / args.status).resolve()
    python = args.python.resolve()
    spec = _load_json(spec_path)
    if spec.get("selection_scope") != "train_validation_only":
        raise ValueError("Matrix spec must be train_validation_only.")
    if spec.get("test_split_read") is not False:
        raise ValueError("Matrix spec must prohibit test reads.")

    jobs_by_name = {str(job["name"]): job for job in spec["jobs"]}
    status: dict[str, Any] = {
        "schema_version": "1.0",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "state": "running",
        "selection_scope": "train_validation_only",
        "test_split_read": False,
        "spec": str(args.spec),
        "jobs": {},
    }
    _write_json(status_path, status)

    completed: dict[str, dict[str, Any]] = {}
    try:
        for job in spec["jobs"]:
            name = str(job["name"])
            output = (root / str(job["output"])).resolve()
            if root not in output.parents:
                raise ValueError(f"Output escapes repository: {output}")
            metrics_path = output / "metrics.json"
            slices_path = output / "validation_risk_slices.json"
            status["jobs"][name] = {"state": "running", "output": str(job["output"])}
            _write_json(status_path, status)

            if not metrics_path.is_file():
                if output.exists():
                    raise RuntimeError(
                        f"Partial output requires manual review before resume: {output}"
                    )
                temporary = output.with_name(output.name + ".partial")
                if temporary.exists():
                    raise RuntimeError(
                        f"Partial temporary output requires manual review: {temporary}"
                    )
                candidate, _ = _job_commands(
                    root=root,
                    python=python,
                    spec=spec,
                    job=job,
                    output=temporary,
                )
                _run(candidate, root=root, dry_run=args.dry_run)
                if args.dry_run:
                    status["jobs"][name] = {"state": "dry_run"}
                    continue
                temporary.rename(output)

            metrics = _validate_metrics(metrics_path, expected_seed=int(job["seed"]))
            if not slices_path.is_file():
                _, slices = _job_commands(
                    root=root,
                    python=python,
                    spec=spec,
                    job=job,
                    output=output,
                )
                _run(slices, root=root, dry_run=args.dry_run)
            if not args.dry_run and not slices_path.is_file():
                raise FileNotFoundError(slices_path)
            completed[name] = metrics
            status["jobs"][name] = {
                "state": "complete",
                "output": str(job["output"]),
                "validation_pr_auc": metrics["validation_metrics"]["pr_auc"],
            }
            _write_json(status_path, status)

        if args.dry_run:
            status["state"] = "dry_run_complete"
            _write_json(status_path, status)
            return

        gates: list[dict[str, Any]] = []
        for pair in spec["paired_gates"]:
            baseline_job = jobs_by_name[str(pair["baseline"])]
            candidate_job = jobs_by_name[str(pair["candidate"])]
            baseline_output = root / str(baseline_job["output"])
            candidate_output = root / str(candidate_job["output"])
            gate_output = candidate_output / "candidate_gate.json"
            if not gate_output.is_file():
                command = [
                    str(python),
                    "scripts/evaluate_novelty_candidate_gate.py",
                    "--baseline-metrics",
                    str((baseline_output / "metrics.json").relative_to(root)),
                    "--candidate-metrics",
                    str((candidate_output / "metrics.json").relative_to(root)),
                    "--baseline-slices",
                    str((baseline_output / "validation_risk_slices.json").relative_to(root)),
                    "--candidate-slices",
                    str((candidate_output / "validation_risk_slices.json").relative_to(root)),
                    "--output",
                    str(gate_output.relative_to(root)),
                ]
                _run(command, root=root, dry_run=False)
            gates.append(_load_json(gate_output))

        rows = [_metric_row(name, payload) for name, payload in completed.items()]
        for name, path in spec["existing_gat_cells"].items():
            payload = _load_json(root / str(path))
            if payload.get("test_split_read") is not False:
                raise ValueError(f"Existing GAT cell may have read test: {path}")
            rows.append(_metric_row(str(name), payload))
        summary = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "selection_scope": "train_validation_only",
            "test_split_read": False,
            "matrix": rows,
            "graphsage_paired_gates": gates,
            "both_graphsage_seed_gates_passed": all(gate["passed"] for gate in gates),
            "test_action": spec["gate_policy"]["test_action"],
        }
        summary_path = root / str(spec["summary_output"])
        _write_json(summary_path, summary)
        status["state"] = "complete"
        status["completed_at_utc"] = datetime.now(UTC).isoformat()
        status["summary"] = str(spec["summary_output"])
        _write_json(status_path, status)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as error:
        status["state"] = "failed"
        status["failed_at_utc"] = datetime.now(UTC).isoformat()
        status["error"] = f"{type(error).__name__}: {error}"
        _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
