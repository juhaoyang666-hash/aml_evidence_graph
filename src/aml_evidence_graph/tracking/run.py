"""Atomic run manifests and structured logs without transaction-record payloads."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fingerprint_path(path: Path) -> dict[str, Any]:
    """Fingerprint a source or artifact path without reading transaction contents into logs."""
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return {
            "kind": "file",
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    if path.is_dir():
        entries = [
            {
                "relative_path": item.relative_to(path).as_posix(),
                "bytes": item.stat().st_size,
            }
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ]
        return {
            "kind": "directory",
            "file_count": len(entries),
            "listing_sha256": _sha256_bytes(
                json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode()
            ),
        }
    raise FileNotFoundError(f"Cannot fingerprint missing path: {path}")


def _package_versions() -> dict[str, str | None]:
    packages = (
        "aml-evidence-graph",
        "catboost",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "torch",
        "torch-geometric",
        "langgraph",
    )
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _git_revision(workdir: Path) -> str | None:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return output.stdout.strip() or None


def _hardware_info() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
    }
    try:
        import torch

        result["torch_cuda_available"] = bool(torch.cuda.is_available())
        result["torch_cuda_version"] = torch.version.cuda
        result["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        result["torch_cuda_available"] = False
        result["torch_cuda_version"] = None
        result["gpu_name"] = None
    return result


@dataclass(frozen=True)
class RunManifest:
    """Reproducibility metadata that intentionally excludes rows and identifiers."""

    schema_version: str
    run_id: str
    created_at_utc: str
    command: str
    random_seed: int
    inputs: dict[str, dict[str, Any]]
    config_fingerprints: dict[str, dict[str, Any]]
    source_revision: str | None
    package_versions: dict[str, str | None]
    hardware: dict[str, Any]
    metadata: dict[str, Any]


def create_run_manifest(
    *,
    output_dir: Path,
    command: str,
    random_seed: int,
    input_paths: dict[str, Path],
    config_paths: dict[str, Path] | None = None,
    metadata: dict[str, Any] | None = None,
    filename: str = "run_manifest.json",
) -> RunManifest:
    """Write an atomic reproducibility manifest beside private run artifacts."""
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("Manifest filename must be a JSON basename.")
    now = datetime.now(UTC)
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:10]}"
    manifest = RunManifest(
        schema_version="1.0",
        run_id=run_id,
        created_at_utc=now.isoformat(),
        command=command,
        random_seed=random_seed,
        inputs={name: fingerprint_path(path) for name, path in input_paths.items()},
        config_fingerprints={
            name: fingerprint_path(path)
            for name, path in (config_paths or {}).items()
        },
        source_revision=_git_revision(output_dir.parent),
        package_versions=_package_versions(),
        hardware=_hardware_info(),
        metadata=metadata or {},
    )
    output_path = output_dir / filename
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return manifest


class JsonLogFormatter(logging.Formatter):
    """Emit structured operational events without serializing raw data objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }
        for field in ("run_id", "alert_id", "as_of_ts", "model_version", "duration_ms"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_structured_logger(
    name: str,
    *,
    run_id: str,
) -> logging.LoggerAdapter[logging.Logger]:
    """Create an isolated JSON logger for a run without duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    return logging.LoggerAdapter(logger, {"run_id": run_id})
