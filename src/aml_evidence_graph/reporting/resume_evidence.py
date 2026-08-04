"""Build a single resume evidence report only from complete run artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from aml_evidence_graph.investigation.llm_review import (
    LLMPublicEvaluation,
    load_public_llm_evaluation,
    validate_public_llm_evaluation,
)


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
    score_path: Path | None = None
    score_column: str | None = None
    expected_test_start_date: str | None = None
    expected_test_end_date: str | None = None
    bootstrap_path: Path | None = None
    bootstrap_component: str | None = None
    minimum_bootstrap_iterations: int | None = Field(default=None, ge=1)
    comparison_group: str | None = None
    comparison_role: Literal["main", "sidecar"] | None = None


class ResumeEvidenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    dataset_disclosure: str
    protocol_disclosure: str
    require_provenance: bool = False
    required_comparison_groups: list[str] = Field(default_factory=list)
    llm_publication_path: Path | None = None
    llm_adjudication_paths: list[Path] = Field(default_factory=list)
    llm_holdout_protocol_path: Path | None = None
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
    source_revision: str | None = None
    input_fingerprints: dict[str, str] = Field(default_factory=dict)
    config_fingerprints: dict[str, str] = Field(default_factory=dict)
    test_rows: int | None = None
    positive_count: int | None = None
    test_start_utc: str | None = None
    test_end_utc: str | None = None
    score_column: str | None = None
    bootstrap_iterations: int | None = None
    comparison_group: str | None = None
    comparison_role: Literal["main", "sidecar"] | None = None


class ResumeComparisonEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparison_group: str
    main_source_id: str
    sidecar_source_id: str
    main_test_pr_auc: float
    sidecar_test_pr_auc: float
    absolute_delta: float
    outcome: Literal["improved", "regressed", "unchanged"]


class ResumeEvidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    public_ready: bool
    dataset_disclosure: str
    protocol_disclosure: str
    evidence: list[ResumeMetricEvidence]
    comparisons: list[ResumeComparisonEvidence] = Field(default_factory=list)
    llm_evaluation: LLMPublicEvaluation | None = None
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


def _fingerprints(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, str] = {}
    for name, value in payload.items():
        if not isinstance(value, dict):
            continue
        fingerprint = value.get("sha256") or value.get("listing_sha256")
        if isinstance(fingerprint, str) and fingerprint:
            result[str(name)] = fingerprint
    return result


def _score_evidence(
    path: Path,
    *,
    score_column: str,
) -> tuple[int, int, str, str]:
    required = {"transaction_id", "event_ts", "is_laundering", score_column}
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema.names)
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError("score file missing columns: " + ", ".join(missing))
    table = pq.read_table(path, columns=["event_ts", "is_laundering", score_column])
    score_values = table[score_column]
    minimum = pc.min(score_values).as_py()
    maximum = pc.max(score_values).as_py()
    null_count = score_values.null_count
    if (
        null_count
        or not isinstance(minimum, int | float)
        or not isinstance(maximum, int | float)
        or not math.isfinite(float(minimum))
        or not math.isfinite(float(maximum))
        or minimum < 0
        or maximum > 1
    ):
        raise ValueError("score column must contain finite probabilities in [0, 1]")
    positive_count = int(pc.sum(table["is_laundering"]).as_py())
    start = pc.min(table["event_ts"]).as_py()
    end = pc.max(table["event_ts"]).as_py()
    return table.num_rows, positive_count, start.isoformat(), end.isoformat()


def _bootstrap_iterations(payload: object, component: str | None) -> int | None:
    candidates: list[tuple[str, int]] = []

    def visit(value: object, path: str) -> None:
        if not isinstance(value, dict):
            return
        iterations = value.get("iterations")
        if isinstance(iterations, int):
            candidates.append((path.lower(), iterations))
        for key, child in value.items():
            visit(child, f"{path}.{key}" if path else str(key))

    visit(payload, "")
    if component:
        matches = [value for path, value in candidates if component.lower() in path]
        if matches:
            return max(matches)
        global_iterations = [value for path, value in candidates if not path]
        return max(global_iterations, default=None)
    return max((value for _, value in candidates), default=None)


def _build_comparisons(
    evidence: list[ResumeMetricEvidence],
) -> list[ResumeComparisonEvidence]:
    grouped: dict[str, dict[str, ResumeMetricEvidence]] = {}
    for item in evidence:
        if item.comparison_group and item.comparison_role:
            grouped.setdefault(item.comparison_group, {})[item.comparison_role] = item
    comparisons: list[ResumeComparisonEvidence] = []
    for group, roles in sorted(grouped.items()):
        if set(roles) != {"main", "sidecar"}:
            continue
        main = roles["main"]
        sidecar = roles["sidecar"]
        delta = sidecar.test_pr_auc - main.test_pr_auc
        outcome: Literal["improved", "regressed", "unchanged"]
        if delta > 1e-12:
            outcome = "improved"
        elif delta < -1e-12:
            outcome = "regressed"
        else:
            outcome = "unchanged"
        comparisons.append(
            ResumeComparisonEvidence(
                comparison_group=group,
                main_source_id=main.source_id,
                sidecar_source_id=sidecar.source_id,
                main_test_pr_auc=main.test_pr_auc,
                sidecar_test_pr_auc=sidecar.test_pr_auc,
                absolute_delta=delta,
                outcome=outcome,
            )
        )
    return comparisons


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
        if metrics.get("run_id") and metrics.get("run_id") != manifest.get("run_id"):
            incomplete.append(f"{source.source_id}:run_id_mismatch")
            continue
        input_fingerprints = _fingerprints(manifest.get("inputs"))
        if spec.require_provenance and (
            not manifest.get("source_revision") or not input_fingerprints
        ):
            incomplete.append(f"{source.source_id}:missing_provenance")
            continue
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
        if (
            not isinstance(pr_auc, int | float)
            or not math.isfinite(float(pr_auc))
            or not 0 <= float(pr_auc) <= 1
        ):
            incomplete.append(f"{source.source_id}:missing_test_pr_auc")
            continue
        run_id = metrics.get("run_id") or manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            incomplete.append(f"{source.source_id}:missing_run_id")
            continue
        roc_auc = selected.get("roc_auc")
        alert_metrics = selected.get("alert_budget_metrics") or selected.get(
            "alert_budgets", {}
        )
        sample_count = selected.get("sample_count")
        positive_count = selected.get("positive_count")
        score_rows: int | None = None
        score_positives: int | None = None
        test_start: str | None = None
        test_end: str | None = None
        if source.score_path is not None:
            if not source.score_column:
                incomplete.append(f"{source.source_id}:missing_score_column_config")
                continue
            score_file = root / source.score_path
            if not score_file.is_file():
                incomplete.append(f"{source.source_id}:missing_score_file")
                continue
            try:
                score_rows, score_positives, test_start, test_end = _score_evidence(
                    score_file, score_column=source.score_column
                )
            except ValueError as error:
                incomplete.append(
                    f"{source.source_id}:invalid_score_file_{str(error).replace(' ', '_')}"
                )
                continue
            if isinstance(sample_count, int | float) and score_rows != int(sample_count):
                incomplete.append(f"{source.source_id}:test_row_mismatch")
                continue
            if (
                isinstance(positive_count, int | float)
                and score_positives != int(positive_count)
            ):
                incomplete.append(f"{source.source_id}:positive_count_mismatch")
                continue
            if (
                source.expected_test_start_date
                and not test_start.startswith(source.expected_test_start_date)
            ):
                incomplete.append(f"{source.source_id}:test_start_mismatch")
                continue
            if (
                source.expected_test_end_date
                and not test_end.startswith(source.expected_test_end_date)
            ):
                incomplete.append(f"{source.source_id}:test_end_mismatch")
                continue
        bootstrap_iterations: int | None = None
        if source.bootstrap_path is not None:
            bootstrap_file = root / source.bootstrap_path
            if not bootstrap_file.is_file():
                incomplete.append(f"{source.source_id}:missing_bootstrap")
                continue
            bootstrap_iterations = _bootstrap_iterations(
                _load_json(bootstrap_file), source.bootstrap_component
            )
            if (
                source.minimum_bootstrap_iterations is not None
                and (
                    bootstrap_iterations is None
                    or bootstrap_iterations < source.minimum_bootstrap_iterations
                )
            ):
                incomplete.append(f"{source.source_id}:insufficient_bootstrap")
                continue
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
                source_revision=(
                    str(manifest["source_revision"])
                    if manifest.get("source_revision")
                    else None
                ),
                input_fingerprints=input_fingerprints,
                config_fingerprints=_fingerprints(manifest.get("config_fingerprints")),
                test_rows=(
                    score_rows
                    if score_rows is not None
                    else int(sample_count)
                    if isinstance(sample_count, int | float)
                    else None
                ),
                positive_count=(
                    score_positives
                    if score_positives is not None
                    else int(positive_count)
                    if isinstance(positive_count, int | float)
                    else None
                ),
                test_start_utc=test_start,
                test_end_utc=test_end,
                score_column=source.score_column,
                bootstrap_iterations=bootstrap_iterations,
                comparison_group=source.comparison_group,
                comparison_role=source.comparison_role,
            )
        )
    comparisons = _build_comparisons(evidence)
    llm_evaluation: LLMPublicEvaluation | None = None
    if spec.llm_publication_path is not None:
        publication_path = root / spec.llm_publication_path
        adjudication_paths = tuple(root / path for path in spec.llm_adjudication_paths)
        if not publication_path.is_file():
            incomplete.append("llm_evaluation:missing_publication")
        elif not adjudication_paths or any(not path.is_file() for path in adjudication_paths):
            incomplete.append("llm_evaluation:missing_adjudication")
        else:
            try:
                llm_evaluation = load_public_llm_evaluation(publication_path)
                protocol_path = (
                    root / spec.llm_holdout_protocol_path
                    if spec.llm_holdout_protocol_path is not None
                    else None
                )
                validate_public_llm_evaluation(
                    llm_evaluation,
                    adjudication_paths,
                    holdout_protocol_path=protocol_path,
                )
            except (ValueError, OSError) as error:
                incomplete.append(
                    "llm_evaluation:invalid_" + str(error).replace(" ", "_")
                )
    completed_groups = {item.comparison_group for item in comparisons}
    for group in spec.required_comparison_groups:
        if group not in completed_groups:
            incomplete.append(f"comparison_{group}:missing_main_or_sidecar")
    required_ids = {source.source_id for source in spec.sources if source.required}
    incomplete_required = [
        item for item in incomplete if item.partition(":")[0] in required_ids
    ]
    incomplete_required.extend(
        item for item in incomplete if item.startswith("comparison_")
    )
    if spec.llm_publication_path is not None:
        incomplete_required.extend(
            item for item in incomplete if item.startswith("llm_evaluation:")
        )
    if incomplete_required and not allow_incomplete:
        raise RuntimeError(
            "Required resume evidence is incomplete: " + ", ".join(incomplete_required)
        )
    return ResumeEvidenceReport(
        public_ready=not incomplete_required,
        dataset_disclosure=spec.dataset_disclosure,
        protocol_disclosure=spec.protocol_disclosure,
        evidence=evidence,
        comparisons=comparisons,
        llm_evaluation=llm_evaluation,
        incomplete_sources=incomplete,
    )


def render_resume_evidence_markdown(report: ResumeEvidenceReport) -> str:
    """Render only values already validated from artifact manifests and metrics."""
    lines = [
        "# 简历证据",
        "",
        f"- Public ready: **{str(report.public_ready).lower()}**",
        f"- Dataset: {report.dataset_disclosure}",
        f"- Protocol: {report.protocol_disclosure}",
        "",
        "## 已验证模型证据",
        "",
        "| 模型/run | 用途 | Test PR-AUC | Test ROC-AUC | 测试行/正例 | Bootstrap | run_id |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in report.evidence:
        roc_auc = f"{item.test_roc_auc:.6f}" if item.test_roc_auc is not None else "—"
        lines.append(
            f"| {item.display_name} | {item.run_purpose} | {item.test_pr_auc:.6f} | "
            f"{roc_auc} | {item.test_rows or '—'}/{item.positive_count or '—'} | "
            f"{item.bootstrap_iterations or '—'} | `{item.run_id}` |"
        )
    if report.comparisons:
        lines.extend(
            [
                "",
                "## 历史基准与当前主线对照",
                "",
                "| 模型组 | 历史基准 PR-AUC | 当前主线 PR-AUC | Δ | 结论 |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for item in report.comparisons:
            lines.append(
                f"| {item.comparison_group} | {item.main_test_pr_auc:.6f} | "
                f"{item.sidecar_test_pr_auc:.6f} | {item.absolute_delta:+.6f} | "
                f"{item.outcome} |"
            )
    if report.evidence:
        lines.extend(
            [
                "",
                "## 协议与追溯",
                "",
                "| 模型/run | 测试时间范围（UTC） | score 列 | source revision | 输入/配置指纹数 |",
                "|---|---|---|---|---:|",
            ]
        )
        for item in report.evidence:
            time_range = (
                f"{item.test_start_utc} → {item.test_end_utc}"
                if item.test_start_utc and item.test_end_utc
                else "—"
            )
            lines.append(
                f"| {item.display_name} | {time_range} | {item.score_column or '—'} | "
                f"`{item.source_revision or '—'}` | "
                f"{len(item.input_fingerprints)}/{len(item.config_fingerprints)} |"
            )
    if report.llm_evaluation is not None:
        llm = report.llm_evaluation
        lines.extend(
            [
                "",
                "## 大模型调查证据",
                "",
                f"- 模型：`{llm.model_name}`；全部人工复核均为项目内部复核。",
                "- v3 开发回归使用原 Golden Set；Holdout 独立于 Prompt 调整并预注册，"
                "但不是外部合规专家验收。",
                "",
                "| 阶段 | Prompt | 外部解析成功率 | 解析后事实门禁 | 人工证据扎根 | "
                "人工总体通过 | P50 / P95 延迟 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        role_names = {
            "frozen_baseline": "首次冻结基线",
            "same_set_development_regression": "同集开发回归",
            "prompt_isolated_project_internal_blind_holdout": "Prompt 隔离 Holdout",
        }
        for stage in llm.stages:
            metrics = stage.metrics
            lines.append(
                f"| {role_names[stage.evaluation_role]}（{stage.case_count}/"
                f"{stage.external_case_count}） | `{stage.prompt_version}` | "
                f"{metrics.external_parse_success_rate:.2%} | "
                f"{metrics.external_fact_validation_pass_rate:.2%} | "
                f"{metrics.human_evidence_grounded_rate:.2%} | "
                f"{metrics.human_overall_pass_rate:.2%} | "
                f"{metrics.latency_p50_ms_all_cases / 1000:.2f}s / "
                f"{metrics.latency_p95_ms_all_cases / 1000:.2f}s |"
            )
        holdout = next(
            stage
            for stage in llm.stages
            if stage.evaluation_role
            == "prompt_isolated_project_internal_blind_holdout"
        )
        if holdout.success_criteria_met is False:
            lines.extend(
                [
                    "",
                    "> Holdout 未通过预注册质量门："
                    + "、".join(holdout.failed_success_criteria)
                    + "。该结果按负结果发布，未用于事后调整 Prompt v3。",
                ]
            )
        if llm.cost_status == "unavailable":
            lines.extend(["", "> 服务方未返回价格元数据，因此不声明金额成本。"])
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
