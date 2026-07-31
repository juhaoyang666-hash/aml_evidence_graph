"""Human-review investigation API; this service does not use an LLM for scoring."""

from __future__ import annotations

import hmac
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from threading import Lock
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, ConfigDict, Field

from aml_evidence_graph.api.private_scoring import PrivateFeaturePartitionScoringService
from aml_evidence_graph.api.services import (
    EvidenceStore,
    HumanReviewRecord,
    InMemoryEvidenceStore,
    InMemoryReviewStore,
    MockPartitionScoringService,
    PartitionScoringService,
    ReviewStore,
    SQLiteEvidenceStore,
)
from aml_evidence_graph.demo.mock_data import build_mock_evidence_package
from aml_evidence_graph.evidence.package import InvestigationReport, RiskEvidencePackage
from aml_evidence_graph.evidence.typology import (
    LocalBM25TypologyRetriever,
    load_typology_documents,
)
from aml_evidence_graph.investigation.audit_store import (
    InMemoryInvestigationAuditStore,
    InvestigationAuditStore,
    SQLiteInvestigationAuditStore,
    persist_state_audit,
)
from aml_evidence_graph.investigation.coordination import (
    LeaseLostError,
    SQLiteThreadLockRegistry,
)
from aml_evidence_graph.investigation.llm import ECNUAnnotationClient, EvidenceAnnotationClient
from aml_evidence_graph.investigation.tools import InvestigationToolRegistry
from aml_evidence_graph.investigation.workflow import run_investigation
from aml_evidence_graph.investigation.workflow_v2 import (
    HumanReviewDecision,
    build_controlled_investigation_graph,
    create_sqlite_checkpointer,
    resume_controlled_investigation,
    start_controlled_investigation,
)
from aml_evidence_graph.settings import Settings

DEMO_HTML = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AML Evidence Graph — Mock Demo</title>
    <style>
      :root { color-scheme: light; font-family: "Segoe UI", "PingFang SC", sans-serif; }
      body { margin: 0; background: #f6f7f9; color: #1b1f24; }
      main { max-width: 920px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }
      h1 { font-size: 1.6rem; margin: 0 0 0.75rem; }
      .banner {
        border-left: 4px solid #b45309; background: #fff7ed; padding: 0.85rem 1rem;
        margin: 0 0 1.25rem; line-height: 1.45;
      }
      button {
        border: 0; background: #0f766e; color: #fff; padding: 0.65rem 1rem;
        border-radius: 6px; cursor: pointer; font-size: 0.95rem;
      }
      button:disabled { opacity: 0.6; cursor: wait; }
      section { margin-top: 1.25rem; }
      h2 { font-size: 1.05rem; margin: 0 0 0.5rem; }
      ul { margin: 0; padding-left: 1.2rem; }
      pre {
        background: #111827; color: #e5e7eb; padding: 1rem; overflow: auto;
        border-radius: 8px; font-size: 0.8rem; line-height: 1.4;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>AML Evidence Graph — 虚构 Demo</h1>
      <div class="banner">
        <strong>边界说明：</strong>本页仅加载内存中的虚构 Evidence Package，不读取
        <code>artifacts/</code>、完整交易或冻结模型分数。演示分数不可当作 SAML-D
        评估指标或业务结论。LLM 不参与评分。
      </div>
      <p>点击下方按钮生成<strong>证据约束</strong>的调查草稿与 SAR 草稿
        （确定性模板，无需外部 LLM）。</p>
      <button id="load">加载 Mock 调查草稿</button>
      <section>
        <h2>摘要</h2>
        <div id="summary">尚未加载。</div>
      </section>
      <section>
        <h2>完整 JSON</h2>
        <pre id="output">No mock data loaded.</pre>
      </section>
    </main>
    <script>
      const summary = document.getElementById("summary");
      const output = document.getElementById("output");
      const button = document.getElementById("load");
      button.onclick = async () => {
        button.disabled = true;
        try {
          const report = await fetch("/demo/cases/mock-alert-0001/draft", {
            method: "POST",
          }).then((response) => {
            if (!response.ok) throw new Error("HTTP " + response.status);
            return response.json();
          });
          const sar = report.sar_draft || {};
          const facts = (report.factual_summary || []).slice(0, 4)
            .map((item) => "<li>" + item + "</li>").join("");
          const pending = (sar.pending_verification || []).slice(0, 3)
            .map((item) => "<li>" + item + "</li>").join("");
          summary.innerHTML =
            "<p><strong>状态：</strong>" + report.status + "</p>" +
            "<p><strong>SAR 标题：</strong>" + (sar.title || "(none)") + "</p>" +
            "<p><strong>事实摘要：</strong></p><ul>" + (facts || "<li>(empty)</li>") + "</ul>" +
            "<p><strong>待核实：</strong></p><ul>" + (pending || "<li>(empty)</li>") + "</ul>" +
            "<p>" + (sar.disclaimer || "") + "</p>";
          output.textContent = JSON.stringify(report, null, 2);
        } catch (error) {
          summary.textContent = "加载失败：" + error;
          output.textContent = String(error);
        } finally {
          button.disabled = false;
        }
      };
    </script>
  </body>
</html>
"""


class ScoreBatchRequest(BaseModel):
    """An opaque, pre-authorized private partition reference."""

    model_config = ConfigDict(extra="forbid")

    partition_ref: str


class ReviewRequest(BaseModel):
    """An internal review outcome, retained as an audit record only."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str
    reviewer_reference: str
    decision: Literal["confirmed", "dismissed", "needs_more_evidence"]
    note: str | None = None


class ControlledInvestigationStartRequest(BaseModel):
    """Optional caller-supplied thread id for a resumable controlled investigation."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str | None = Field(default=None, min_length=1, max_length=128)


class _ThreadLockRegistry:
    """Serialize one thread's mutations inside a single API process."""

    def __init__(self) -> None:
        self._guard = Lock()
        self._entries: dict[str, tuple[Lock, int]] = {}

    @contextmanager
    def hold(self, thread_id: str) -> Iterator[None]:
        with self._guard:
            lock, references = self._entries.get(thread_id, (Lock(), 0))
            self._entries[thread_id] = (lock, references + 1)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._guard:
                current_lock, references = self._entries[thread_id]
                if references == 1:
                    del self._entries[thread_id]
                else:
                    self._entries[thread_id] = (current_lock, references - 1)


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _interrupt_values(
    *,
    state: dict[str, object] | None = None,
    snapshot: object = None,
) -> list[object]:
    values: list[object] = []
    if state is not None:
        for item in state.get("__interrupt__", []):
            values.append(getattr(item, "value", item))
    if snapshot is not None:
        for task in getattr(snapshot, "tasks", ()):
            for item in getattr(task, "interrupts", ()):
                values.append(getattr(item, "value", item))
    return values


def _controlled_api_response(
    thread_id: str,
    state: dict[str, object],
    *,
    interrupts: list[object],
    idempotent_replay: bool = False,
) -> dict[str, object]:
    final_status = state.get("final_status")
    status = "completed" if final_status else "awaiting_human_review" if interrupts else "running"
    return {
        "thread_id": thread_id,
        "status": status,
        "final_status": final_status,
        "report": state.get("report"),
        "tool_calls": state.get("tool_calls", []),
        "audit_events": state.get("audit_events", []),
        "review_prompt": interrupts[0] if interrupts else None,
        "idempotent_replay": idempotent_replay,
    }


def create_app(
    retriever: LocalBM25TypologyRetriever,
    *,
    annotator: EvidenceAnnotationClient | None = None,
    evidence_store: EvidenceStore | None = None,
    scoring_service: PartitionScoringService | None = None,
    review_store: ReviewStore | None = None,
    internal_api_token: str | None = None,
    controlled_checkpointer: object | None = None,
    controlled_audit_store: InvestigationAuditStore | None = None,
    thread_lock_registry: object | None = None,
) -> FastAPI:
    """Create an API bound to a local corpus; LLM use is annotation-only."""
    app = FastAPI(
        title="AML Evidence Graph",
        version="0.1.0",
        description=(
            "Evidence-bound investigation drafts. "
            "Risk scores must be produced by approved model inference upstream."
        ),
    )

    store = evidence_store or InMemoryEvidenceStore()
    mock_evidence = build_mock_evidence_package()
    store.put(mock_evidence)
    scorer = scoring_service or MockPartitionScoringService(store, mock_evidence)
    reviews = review_store or InMemoryReviewStore()
    checkpointer = controlled_checkpointer or InMemorySaver()
    investigation_audits = controlled_audit_store or InMemoryInvestigationAuditStore()
    thread_locks = thread_lock_registry or _ThreadLockRegistry()
    controlled_graph = build_controlled_investigation_graph(
        InvestigationToolRegistry(retriever),
        retriever=retriever,
        annotator=annotator,
        checkpointer=checkpointer,
    )
    if isinstance(scorer, PrivateFeaturePartitionScoringService) and internal_api_token is None:
        raise ValueError("Private feature scoring requires an internal API token.")

    @app.exception_handler(LeaseLostError)
    def handle_lease_lost(request: Request, error: LeaseLostError) -> JSONResponse:
        """A lost lease makes the mutation unconfirmed, so answer retryable, not 500."""
        return JSONResponse(status_code=503, content={"detail": str(error)})

    def require_internal_token(
        x_aml_internal_token: str | None = Header(default=None),
    ) -> None:
        if internal_api_token is None:
            return
        if x_aml_internal_token is None or not hmac.compare_digest(
            x_aml_internal_token,
            internal_api_token,
        ):
            raise HTTPException(status_code=401, detail="Internal API authorization failed.")

    @app.get("/health")
    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "llm_scoring": "disabled",
            "llm_annotation": "enabled" if annotator is not None else "disabled",
        }

    @app.get("/demo", response_class=HTMLResponse)
    def demo() -> str:
        return DEMO_HTML

    @app.get("/demo/evidence", response_model=RiskEvidencePackage)
    def demo_evidence() -> RiskEvidencePackage:
        return mock_evidence

    @app.get("/demo/cases/{demo_case_id}", response_model=RiskEvidencePackage)
    def demo_case(demo_case_id: str) -> RiskEvidencePackage:
        if demo_case_id != mock_evidence.alert_id:
            raise HTTPException(status_code=404, detail="Unknown mock case.")
        return mock_evidence

    @app.post(
        "/demo/cases/{demo_case_id}/draft",
        response_model=InvestigationReport,
    )
    def demo_draft(demo_case_id: str) -> InvestigationReport:
        if demo_case_id != mock_evidence.alert_id:
            raise HTTPException(status_code=404, detail="Unknown mock case.")
        return run_investigation(mock_evidence, retriever=retriever, annotator=None)

    @app.post("/v1/score/batch", dependencies=[Depends(require_internal_token)])
    def score_batch(request: ScoreBatchRequest) -> dict[str, object]:
        try:
            result = scorer.score_partition(request.partition_ref)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {
            "partition_ref": result.partition_ref,
            "alert_ids": result.alert_ids,
            "model_version": result.model_version,
        }

    @app.get(
        "/v1/evidence/{alert_id}",
        response_model=RiskEvidencePackage,
        dependencies=[Depends(require_internal_token)],
    )
    def get_evidence(alert_id: str) -> RiskEvidencePackage:
        evidence = store.get(alert_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail="Unknown alert reference.")
        return evidence

    @app.post(
        "/v1/investigations/{alert_id}/draft",
        response_model=InvestigationReport,
        dependencies=[Depends(require_internal_token)],
    )
    def draft_investigation(alert_id: str) -> InvestigationReport:
        evidence = store.get(alert_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail="Unknown alert reference.")
        return run_investigation(
            evidence,
            retriever=retriever,
            annotator=annotator,
        )

    @app.post("/v1/reviews", dependencies=[Depends(require_internal_token)])
    def record_review(request: ReviewRequest) -> dict[str, object]:
        if store.get(request.alert_id) is None:
            raise HTTPException(status_code=404, detail="Unknown alert reference.")
        record = reviews.append(
            HumanReviewRecord(
                review_id=f"review-{uuid.uuid4().hex}",
                alert_id=request.alert_id,
                reviewer_reference=request.reviewer_reference,
                decision=request.decision,
                note=request.note,
                submitted_at=datetime.now(UTC),
            )
        )
        return {
            "review_id": record.review_id,
            "alert_id": record.alert_id,
            "decision": record.decision,
            "submitted_at": record.submitted_at,
            "model_update": "not_triggered",
        }

    @app.post(
        "/v1/controlled-investigations/{alert_id}",
        dependencies=[Depends(require_internal_token)],
    )
    def start_controlled_case(
        alert_id: str,
        request: ControlledInvestigationStartRequest,
    ) -> dict[str, object]:
        evidence = store.get(alert_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail="Unknown alert reference.")
        thread_id = request.thread_id or f"investigation-{uuid.uuid4().hex}"
        with thread_locks.hold(thread_id):
            snapshot = controlled_graph.get_state(_thread_config(thread_id))
            if snapshot.values:
                raise HTTPException(status_code=409, detail="Investigation thread already exists.")
            state = start_controlled_investigation(
                controlled_graph,
                evidence,
                thread_id=thread_id,
            )
            persist_state_audit(investigation_audits, thread_id=thread_id, state=state)
        return _controlled_api_response(
            thread_id,
            state,
            interrupts=_interrupt_values(state=state),
        )

    @app.get(
        "/v1/controlled-investigations/{thread_id}",
        dependencies=[Depends(require_internal_token)],
    )
    def get_controlled_case(thread_id: str) -> dict[str, object]:
        with thread_locks.hold(thread_id):
            snapshot = controlled_graph.get_state(_thread_config(thread_id))
            if not snapshot.values:
                raise HTTPException(status_code=404, detail="Unknown investigation thread.")
            state = dict(snapshot.values)
            interrupts = _interrupt_values(snapshot=snapshot)
        return _controlled_api_response(
            thread_id,
            state,
            interrupts=interrupts,
        )

    @app.post(
        "/v1/controlled-investigations/{thread_id}/review",
        dependencies=[Depends(require_internal_token)],
    )
    def review_controlled_case(
        thread_id: str,
        decision: HumanReviewDecision,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, object]:
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not idempotency_key or len(idempotency_key) > 128:
                raise HTTPException(
                    status_code=400,
                    detail="Idempotency-Key must contain 1 to 128 characters.",
                )
        replayed = False
        with thread_locks.hold(thread_id):
            snapshot = controlled_graph.get_state(_thread_config(thread_id))
            if not snapshot.values:
                raise HTTPException(status_code=404, detail="Unknown investigation thread.")
            if snapshot.values.get("final_status") is not None:
                if (
                    idempotency_key is not None
                    and snapshot.values.get("review_idempotency_key") == idempotency_key
                ):
                    stored = snapshot.values.get("review_decision")
                    if stored != decision.model_dump(mode="json"):
                        raise HTTPException(
                            status_code=409,
                            detail="Idempotency-Key was reused with a different review request.",
                        )
                    state = dict(snapshot.values)
                    replayed = True
                else:
                    raise HTTPException(
                        status_code=409,
                        detail="Investigation thread is already final.",
                    )
            else:
                if not _interrupt_values(snapshot=snapshot):
                    raise HTTPException(
                        status_code=409,
                        detail="Investigation is not awaiting review.",
                    )
                state = resume_controlled_investigation(
                    controlled_graph,
                    decision,
                    thread_id=thread_id,
                    idempotency_key=idempotency_key,
                )
                persist_state_audit(investigation_audits, thread_id=thread_id, state=state)
        return _controlled_api_response(
            thread_id,
            state,
            interrupts=_interrupt_values(state=state),
            idempotent_replay=replayed,
        )

    return app


def create_default_app() -> FastAPI:
    """Factory for Uvicorn; Typology retrieval stays local to the deployment."""
    settings = Settings()
    settings.validate_agent_storage_separation()
    retriever = LocalBM25TypologyRetriever(
        load_typology_documents(settings.typology_root)
    )
    annotator = ECNUAnnotationClient.from_settings(settings) if settings.llm_enabled else None
    controlled_checkpointer = (
        create_sqlite_checkpointer(settings.agent_checkpoint_path)
        if settings.agent_checkpoint_path is not None
        else None
    )
    controlled_audit_store = (
        SQLiteInvestigationAuditStore(settings.agent_audit_path)
        if settings.agent_audit_path is not None
        else None
    )
    evidence_store = (
        SQLiteEvidenceStore(settings.evidence_store_path)
        if settings.evidence_store_path is not None
        else InMemoryEvidenceStore()
    )
    thread_lock_registry = (
        SQLiteThreadLockRegistry(settings.agent_coordination_path)
        if settings.agent_coordination_path is not None
        else None
    )
    if (settings.feature_root is None) != (settings.table_model_dir is None):
        raise RuntimeError(
            "AML_FEATURE_ROOT and AML_TABLE_MODEL_DIR must be configured together."
        )
    if settings.feature_root is None:
        return create_app(
            retriever,
            annotator=annotator,
            evidence_store=evidence_store,
            internal_api_token=(
                settings.internal_api_token.get_secret_value()
                if settings.internal_api_token is not None
                else None
            ),
            controlled_checkpointer=controlled_checkpointer,
            controlled_audit_store=controlled_audit_store,
            thread_lock_registry=thread_lock_registry,
        )
    internal_api_token = settings.require_internal_api_token()
    model_version = settings.require_model_version()
    scorer = PrivateFeaturePartitionScoringService(
        feature_root=settings.feature_root,
        table_model_dir=settings.table_model_dir,
        evidence_store=evidence_store,
        alert_threshold=settings.alert_threshold,
        source_version=model_version,
        graphsage_artifact_path=settings.graphsage_model_path,
        fusion_dir=settings.fusion_dir,
        graphsage_device=settings.graphsage_device,
    )
    return create_app(
        retriever,
        annotator=annotator,
        evidence_store=evidence_store,
        scoring_service=scorer,
        internal_api_token=internal_api_token,
        controlled_checkpointer=controlled_checkpointer,
        controlled_audit_store=controlled_audit_store,
        thread_lock_registry=thread_lock_registry,
    )


def main() -> None:
    """Run the local investigation API."""
    import uvicorn

    uvicorn.run(
        create_default_app(),
        host="127.0.0.1",
        port=8000,
    )
