"""Aggregate local serving benchmark artifacts into an auditable Chinese report."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ServingBenchmarkSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    display_name: str
    artifact_dir: Path
    required: bool = True


class ServingBenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    hardware_disclosure: str
    sources: list[ServingBenchmarkSource]


class ServingBenchmarkEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    display_name: str
    method: str
    path: str
    requests: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    error_rate: float = Field(ge=0, le=1)
    throughput_per_second: float = Field(ge=0)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    latency_p99_ms: float = Field(ge=0)


class ServingBenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    complete: bool
    scope: str = "Local reproducible observation; not a production SLA."
    hardware_disclosure: str
    evidence: list[ServingBenchmarkEvidence]
    incomplete_sources: list[str]


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def build_serving_benchmark_report(
    spec: ServingBenchmarkSpec,
    *,
    root: Path,
) -> ServingBenchmarkReport:
    evidence: list[ServingBenchmarkEvidence] = []
    incomplete: list[str] = []
    for source in spec.sources:
        metrics_path = root / source.artifact_dir / "metrics.json"
        if not metrics_path.is_file():
            incomplete.append(f"{source.source_id}:missing_metrics")
            continue
        payload = _load_object(metrics_path)
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            incomplete.append(f"{source.source_id}:missing_summary")
            continue
        try:
            evidence.append(
                ServingBenchmarkEvidence(
                    source_id=source.source_id,
                    display_name=source.display_name,
                    method=str(payload["method"]),
                    path=str(payload["path"]),
                    requests=int(summary["requests"]),
                    concurrency=int(payload["concurrency"]),
                    error_rate=float(summary["error_rate"]),
                    throughput_per_second=float(summary["throughput_per_second"]),
                    latency_p50_ms=float(summary["latency_p50_ms"]),
                    latency_p95_ms=float(summary["latency_p95_ms"]),
                    latency_p99_ms=float(summary["latency_p99_ms"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            incomplete.append(f"{source.source_id}:invalid_metrics_{type(error).__name__}")
    required = {source.source_id for source in spec.sources if source.required}
    incomplete_required = [
        item for item in incomplete if item.partition(":")[0] in required
    ]
    return ServingBenchmarkReport(
        complete=not incomplete_required,
        hardware_disclosure=spec.hardware_disclosure,
        evidence=evidence,
        incomplete_sources=incomplete,
    )


def render_serving_benchmark_markdown(report: ServingBenchmarkReport) -> str:
    lines = [
        "# 服务性能基准",
        "",
        "> 本报告仅表示固定本机环境下的可复现实测，不是生产 SLA。",
        "",
        f"- 完整状态：**{str(report.complete).lower()}**",
        f"- 硬件边界：{report.hardware_disclosure}",
        "",
        "| 路径 | 请求/并发 | 错误率 | 吞吐 req/s | p50 ms | p95 ms | p99 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.evidence:
        lines.append(
            f"| {item.display_name} (`{item.method} {item.path}`) | "
            f"{item.requests}/{item.concurrency} | {item.error_rate:.2%} | "
            f"{item.throughput_per_second:.2f} | {item.latency_p50_ms:.2f} | "
            f"{item.latency_p95_ms:.2f} | {item.latency_p99_ms:.2f} |"
        )
    if report.incomplete_sources:
        lines.extend(["", "## 尚缺基准", ""])
        lines.extend(f"- `{item}`" for item in report.incomplete_sources)
    lines.extend(
        [
            "",
            "## 解读边界",
            "",
            "- Mock 路径不加载冻结模型，不能代表模型推理延迟。",
            "- `完整状态=false` 时不得声明完整服务性能；未完成来源以上方清单为准。",
            "- 报告不包含真实交易、账户标识、Token 或外部服务密钥。",
        ]
    )
    return "\n".join(lines) + "\n"
