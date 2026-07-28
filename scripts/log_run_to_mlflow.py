"""Mirror one completed aggregate-only artifact directory into local MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from aml_evidence_graph.tracking.mlflow_adapter import log_completed_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--experiment", default="aml-evidence-graph")
    parser.add_argument("--tracking-uri", default="sqlite:///artifacts/mlflow.db")
    args = parser.parse_args()
    run_id = log_completed_run(
        args.artifact_dir,
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
    )
    print(f"MLflow run: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
