from __future__ import annotations

import json
from pathlib import Path

import pytest

from aml_evidence_graph.reporting.resume_evidence import (
    ResumeEvidenceSourceSpec,
    ResumeEvidenceSpec,
    build_resume_evidence,
    render_resume_evidence_markdown,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resume_evidence_requires_metrics_and_manifest(tmp_path: Path) -> None:
    spec = ResumeEvidenceSpec(
        dataset_disclosure="synthetic",
        protocol_disclosure="time-out",
        sources=[
            ResumeEvidenceSourceSpec(
                source_id="table",
                display_name="Table",
                artifact_dir=Path("artifacts/table"),
                component="catboost",
            )
        ],
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        build_resume_evidence(spec, root=tmp_path)


def test_resume_evidence_renders_only_verified_values(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts/table"
    _write_json(
        artifact / "run_manifest.json",
        {"run_id": "run-1", "run_purpose": "full"},
    )
    _write_json(
        artifact / "metrics.json",
        {
            "run_id": "run-1",
            "test_metrics": {"catboost": {"pr_auc": 0.8, "roc_auc": 0.9}},
        },
    )
    spec = ResumeEvidenceSpec(
        dataset_disclosure="synthetic",
        protocol_disclosure="time-out",
        sources=[
            ResumeEvidenceSourceSpec(
                source_id="table",
                display_name="Table",
                artifact_dir=Path("artifacts/table"),
                component="catboost",
            )
        ],
    )

    report = build_resume_evidence(spec, root=tmp_path)
    markdown = render_resume_evidence_markdown(report)

    assert report.public_ready
    assert report.evidence[0].test_pr_auc == 0.8
    assert "run-1" in markdown
    assert "0.800000" in markdown


def test_resume_evidence_rejects_smoke_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts/table"
    _write_json(
        artifact / "run_manifest.json",
        {"run_id": "smoke-1", "run_purpose": "smoke"},
    )
    _write_json(
        artifact / "metrics.json",
        {
            "run_id": "smoke-1",
            "test_metrics": {"catboost": {"pr_auc": 0.99, "sample_count": 20}},
        },
    )
    spec = ResumeEvidenceSpec(
        dataset_disclosure="synthetic",
        protocol_disclosure="time-out",
        sources=[
            ResumeEvidenceSourceSpec(
                source_id="table",
                display_name="Table",
                artifact_dir=Path("artifacts/table"),
                component="catboost",
            )
        ],
    )

    with pytest.raises(RuntimeError, match="run_purpose_smoke"):
        build_resume_evidence(spec, root=tmp_path)


def test_legacy_manifest_requires_large_test_population_gate(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts/table"
    _write_json(artifact / "run_manifest.json", {"run_id": "legacy-1"})
    _write_json(
        artifact / "metrics.json",
        {
            "run_id": "legacy-1",
            "test_metrics": {
                "catboost": {"pr_auc": 0.8, "sample_count": 1_500_000}
            },
        },
    )
    spec = ResumeEvidenceSpec(
        dataset_disclosure="synthetic",
        protocol_disclosure="time-out",
        sources=[
            ResumeEvidenceSourceSpec(
                source_id="table",
                display_name="Table",
                artifact_dir=Path("artifacts/table"),
                component="catboost",
                legacy_minimum_test_rows=1_000_000,
            )
        ],
    )

    report = build_resume_evidence(spec, root=tmp_path)

    assert report.public_ready
    assert report.evidence[0].run_purpose == "legacy_full_by_row_gate"
