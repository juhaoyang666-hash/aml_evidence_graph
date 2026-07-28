"""Build a single resume evidence report only from complete run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResumeEvidenceSourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    display_name: str
    artifact_dir: Path
    component: str | None = None
    required: bool = True
    pipeline_status: Path | None = None
    required_pipeline_state: str = "complete"
    expected_run_purpose: Literal["full"] = "full"
    legacy_minimum_test_rows: int | None = Field(default=None, ge=1)


class ResumeEvidenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    dataset_disclosure: str
    protocol_disclosure: str
    sources: list[ResumeEvidenceSourceSpec]


class ResumeMetricEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    display_name: str
    artifact_dir: str
    run_id: str
    run_purpose: str
    test_pr_auc: float
    test_roc_auc: float | None = None
    alert_budget_metrics: dict[str, object] = Field(default_factory=dict)


class ResumeEvidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    public_ready: bool
    dataset_disclosure: str
    protocol_disclosure: str
    evidence: list[ResumeMetricEvidence]
    incomplete_sources: list[str]


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _selected_test_metrics(
    metrics: dict[str, object],
    component: str | None,
) -> dict[str, object]:
    test_metrics = metrics.get("test_metrics")
    if not isinstance(test_metrics, dict):
        raise ValueError("metrics.json has no test_metrics object.")
    if component is None:
        return test_metrics
    selected = test_metrics.get(component)
    if not isinstance(selected, dict):
        raise ValueError(f"test_metrics has no component {component!r}.")
    return selected


def build_resume_evidence(
    spec: ResumeEvidenceSpec,
    *,
    root: Path,
    allow_incomplete: bool = False,
) -> ResumeEvidenceReport:
    """Read manifests/metrics and reject smoke, running, or partial artifacts."""
    evidence: list[ResumeMetricEvidence] = []
    incomplete: list[str] = []
    for source in spec.sources:
        artifact_dir = root / source.artifact_dir
        metrics_path = artifact_dir / "metrics.json"
        manifest_path = artifact_dir / "run_manifest.json"
        if source.pipeline_status is not None:
            status_path = root / source.pipeline_status
            if not status_path.is_file():
                incomplete.append(f"{source.source_id}:missing_pipeline_status")
                continue
            status = _load_json(status_path)
            if status.get("current_state") != source.required_pipeline_state:
                incomplete.append(
                    f"{source.source_id}:pipeline_{status.get('current_state', 'unknown')}"
                )
                continue
        if not metrics_path.is_file() or not manifest_path.is_file():
            incomplete.append(f"{source.source_id}:missing_metrics_or_manifest")
            continue
        metrics = _load_json(metrics_path)
        manifest = _load_json(manifest_path)
        selected = _selected_test_metrics(metrics, source.component)
        manifest_purpose = manifest.get("run_purpose")
        if manifest_purpose is None:
            sample_count = metrics.get("test_rows") or selected.get("sample_count")
            if (
                source.legacy_minimum_test_rows is None
                or not isinstance(sample_count, int | float)
                or int(sample_count) < source.legacy_minimum_test_rows
            ):
                incomplete.append(f"{source.source_id}:missing_run_purpose")
                continue
            run_purpose = "legacy_full_by_row_gate"
        elif manifest_purpose != source.expected_run_purpose:
            incomplete.append(f"{source.source_id}:run_purpose_{manifest_purpose}")
            continue
        else:
            run_purpose = str(manifest_purpose)
        pr_auc = selected.get("pr_auc")
        if not isinstance(pr_auc, int | float):
            incomplete.append(f"{source.source_id}:missing_test_pr_auc")
            continue
        run_id = metrics.get("run_id") or manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            incomplete.append(f"{source.source_id}:missing_run_id")
            continue
        roc_auc = selected.get("roc_auc")
        alert_metrics = selected.get("alert_budget_metrics", {})
        evidence.append(
            ResumeMetricEvidence(
                source_id=source.source_id,
                display_name=source.display_name,
                artifact_dir=source.artifact_dir.as_posix(),
                run_id=run_id,
                run_purpose=run_purpose,
                test_pr_auc=float(pr_auc),
                test_roc_auc=float(roc_auc) if isinstance(roc_auc, int | float) else None,
                alert_budget_metrics=(
                    alert_metrics if isinstance(alert_metrics, dict) else {}
                ),
            )
        )
    required_ids = {source.source_id for source in spec.sources if source.required}
    incomplete_required = [
        item for item in incomplete if item.partition(":")[0] in required_ids
    ]
    if incomplete_required and not allow_incomplete:
        raise RuntimeError(
            "Required resume evidence is incomplete: " + ", ".join(incomplete_required)
        )
    return ResumeEvidenceReport(
        public_ready=not incomplete_required,
        dataset_disclosure=spec.dataset_disclosure,
        protocol_disclosure=spec.protocol_disclosure,
        evidence=evidence,
        incomplete_sources=incomplete,
    )


def render_resume_evidence_markdown(report: ResumeEvidenceReport) -> str:
    """Render only values already validated from artifact manifests and metrics."""
    lines = [
        "# Resume Evidence",
        "",
        f"- Public ready: **{str(report.public_ready).lower()}**",
        f"- Dataset: {report.dataset_disclosure}",
        f"- Protocol: {report.protocol_disclosure}",
        "",
        "## Verified model evidence",
        "",
        "| Model/run | Purpose | Test PR-AUC | Test ROC-AUC | run_id | Artifact |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in report.evidence:
        roc_auc = f"{item.test_roc_auc:.6f}" if item.test_roc_auc is not None else "—"
        lines.append(
            f"| {item.display_name} | {item.run_purpose} | {item.test_pr_auc:.6f} | "
            f"{roc_auc} | "
            f"`{item.run_id}` | `{item.artifact_dir}` |"
        )
    if report.incomplete_sources:
        lines.extend(["", "## Incomplete sources", ""])
        lines.extend(f"- `{item}`" for item in report.incomplete_sources)
    lines.extend(
        [
            "",
            "> Metrics are copied from completed local artifacts; smoke and running "
            "artifacts are excluded.",
        ]
    )
    return "\n".join(lines) + "\n"
