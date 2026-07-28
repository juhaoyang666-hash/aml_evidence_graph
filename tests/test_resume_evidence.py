from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
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


def _write_scores(path: Path, score_column: str, scores: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "transaction_id": [f"txn-{index}" for index in range(len(scores))],
                "event_ts": [
                    datetime(2023, 7, 1, 0, 0, index + 1, tzinfo=UTC)
                    for index in range(len(scores))
                ],
                "is_laundering": [False, True][: len(scores)],
                score_column: scores,
            }
        ),
        path,
    )


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


def test_resume_evidence_validates_protocol_and_compares_sidecar(tmp_path: Path) -> None:
    sources: list[ResumeEvidenceSourceSpec] = []
    for role, pr_auc, scores in (
        ("main", 0.80, [0.1, 0.8]),
        ("sidecar", 0.85, [0.2, 0.9]),
    ):
        artifact = tmp_path / f"artifacts/{role}"
        _write_json(
            artifact / "run_manifest.json",
            {
                "run_id": f"run-{role}",
                "run_purpose": "full",
                "source_revision": "abc123",
                "inputs": {"features": {"listing_sha256": f"sha-{role}"}},
                "config_fingerprints": {"model": {"sha256": "config-sha"}},
            },
        )
        _write_json(
            artifact / "metrics.json",
            {
                "run_id": f"run-{role}",
                "test_metrics": {
                    "catboost": {
                        "pr_auc": pr_auc,
                        "roc_auc": 0.9,
                        "sample_count": 2,
                        "positive_count": 1,
                    }
                },
            },
        )
        _write_scores(artifact / "scores.parquet", "catboost", scores)
        _write_json(
            artifact / "bootstrap.json",
            {"pr_auc": {"iterations": 200}},
        )
        sources.append(
            ResumeEvidenceSourceSpec(
                source_id=role,
                display_name=role,
                artifact_dir=Path(f"artifacts/{role}"),
                component="catboost",
                score_path=Path(f"artifacts/{role}/scores.parquet"),
                score_column="catboost",
                expected_test_start_date="2023-07-01",
                expected_test_end_date="2023-07-01",
                bootstrap_path=Path(f"artifacts/{role}/bootstrap.json"),
                minimum_bootstrap_iterations=200,
                comparison_group="CatBoost",
                comparison_role=role,
            )
        )
    spec = ResumeEvidenceSpec(
        dataset_disclosure="synthetic",
        protocol_disclosure="time-out",
        require_provenance=True,
        sources=sources,
    )

    report = build_resume_evidence(spec, root=tmp_path)
    markdown = render_resume_evidence_markdown(report)

    assert report.public_ready
    assert report.evidence[0].test_rows == 2
    assert report.evidence[0].positive_count == 1
    assert report.evidence[0].bootstrap_iterations == 200
    assert report.evidence[0].input_fingerprints == {"features": "sha-main"}
    assert report.comparisons[0].absolute_delta == pytest.approx(0.05)
    assert report.comparisons[0].outcome == "improved"
    assert "主线与 FE v2 sidecar 对照" in markdown


def test_resume_evidence_rejects_score_population_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts/table"
    _write_json(
        artifact / "run_manifest.json",
        {"run_id": "run-1", "run_purpose": "full"},
    )
    _write_json(
        artifact / "metrics.json",
        {
            "run_id": "run-1",
            "test_metrics": {
                "catboost": {
                    "pr_auc": 0.8,
                    "sample_count": 3,
                    "positive_count": 1,
                }
            },
        },
    )
    _write_scores(artifact / "scores.parquet", "catboost", [0.1, 0.8])
    spec = ResumeEvidenceSpec(
        dataset_disclosure="synthetic",
        protocol_disclosure="time-out",
        sources=[
            ResumeEvidenceSourceSpec(
                source_id="table",
                display_name="Table",
                artifact_dir=Path("artifacts/table"),
                component="catboost",
                score_path=Path("artifacts/table/scores.parquet"),
                score_column="catboost",
            )
        ],
    )

    with pytest.raises(RuntimeError, match="test_row_mismatch"):
        build_resume_evidence(spec, root=tmp_path)
