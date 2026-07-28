"""Report optional career-extension capabilities without importing heavy packages."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys

PACKAGES = {
    "agent_persistence": "langgraph-checkpoint-sqlite",
    "semantic_retrieval": "sentence-transformers",
    "mlflow": "mlflow",
    "spark": "pyspark",
}


def _package_status(distribution: str) -> dict[str, object]:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}
    return {"installed": True, "version": version}


def _cuda_status() -> dict[str, object]:
    if importlib.util.find_spec("torch") is None:
        return {"torch_installed": False, "cuda_available": False}
    import torch

    return {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


def _java_status() -> dict[str, object]:
    executable = shutil.which("java")
    if executable is None:
        return {"installed": False, "executable": None}
    completed = subprocess.run(
        [executable, "-version"], capture_output=True, text=True, check=False
    )
    first_line = (completed.stderr or completed.stdout).splitlines()
    return {
        "installed": completed.returncode == 0,
        "executable": executable,
        "version": first_line[0] if first_line else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail unless SQLite checkpoint support is installed and CUDA is available.",
    )
    args = parser.parse_args()
    report = {
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "packages": {name: _package_status(package) for name, package in PACKAGES.items()},
        "cuda": _cuda_status(),
        "java": _java_status(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict:
        agent_ready = report["packages"]["agent_persistence"]["installed"]
        cuda_ready = report["cuda"]["cuda_available"]
        return 0 if agent_ready and cuda_ready else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
