"""Human-review investigation API; this service does not use an LLM for scoring."""

from __future__ import annotations

import hmac
import uuid
from datetime import UTC, datetime
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from aml_evidence_graph.api.private_scoring import PrivateFeaturePartitionScoringService
from aml_evidence_graph.api.services import (
    EvidenceStore,
    HumanReviewRecord,
    InMemoryEvidenceStore,
    InMemoryReviewStore,
    MockPartitionScoringService,
    PartitionScoringService,
    ReviewStore,
)
from aml_evidence_graph.demo.mock_data import build_mock_evidence_package
from aml_evidence_graph.evidence.package import InvestigationReport, RiskEvidencePackage
from aml_evidence_graph.evidence.typology import (
    LocalBM25TypologyRetriever,
    load_typology_documents,
)
from aml_evidence_graph.investigation.llm import ECNUAnnotationClient, EvidenceAnnotationClient
from aml_evidence_graph.investigation.workflow import run_investigation
from aml_evidence_graph.settings import Settings

DEMO_HTML = """
<!doctype html>
<html lang="en">
  <head><meta charset="utf-8"><title>AML Evidence Graph — Mock Demo</title></head>
  <body>
    <main>
      <h1>AML Evidence Graph — fictional demonstration</h1>
      <p>This page uses only synthetic evidence.</p>
      <p>It does not load private transactions or scores.</p>
      <button id="load">Load mock review draft</button>
      <pre id="output">No mock data loaded.</pre>
    </main>
    <script>
      document.getElementById("load").onclick = async () => {
        const report = await fetch("/demo/cases/mock-alert-0001/draft", {
          method: "POST",
        }).then((response) => response.json());
        document.getElementById("output").textContent = JSON.stringify(report, null, 2);
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


def create_app(
    retriever: LocalBM25TypologyRetriever,
    *,
    annotator: EvidenceAnnotationClient | None = None,
    evidence_store: EvidenceStore | None = None,
    scoring_service: PartitionScoringService | None = None,
    review_store: ReviewStore | None = None,
    internal_api_token: str | None = None,
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
    if isinstance(scorer, PrivateFeaturePartitionScoringService) and internal_api_token is None:
        raise ValueError("Private feature scoring requires an internal API token.")

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

    return app


def create_default_app() -> FastAPI:
    """Factory for Uvicorn; Typology retrieval stays local to the deployment."""
    settings = Settings()
    retriever = LocalBM25TypologyRetriever(
        load_typology_documents(settings.typology_root)
    )
    annotator = ECNUAnnotationClient.from_settings(settings) if settings.llm_enabled else None
    if (settings.feature_root is None) != (settings.table_model_dir is None):
        raise RuntimeError(
            "AML_FEATURE_ROOT and AML_TABLE_MODEL_DIR must be configured together."
        )
    if settings.feature_root is None:
        return create_app(
            retriever,
            annotator=annotator,
            internal_api_token=(
                settings.internal_api_token.get_secret_value()
                if settings.internal_api_token is not None
                else None
            ),
        )
    internal_api_token = settings.require_internal_api_token()
    model_version = settings.require_model_version()
    store = InMemoryEvidenceStore()
    scorer = PrivateFeaturePartitionScoringService(
        feature_root=settings.feature_root,
        table_model_dir=settings.table_model_dir,
        evidence_store=store,
        alert_threshold=settings.alert_threshold,
        source_version=model_version,
        graphsage_artifact_path=settings.graphsage_model_path,
        fusion_dir=settings.fusion_dir,
        graphsage_device=settings.graphsage_device,
    )
    return create_app(
        retriever,
        annotator=annotator,
        evidence_store=store,
        scoring_service=scorer,
        internal_api_token=internal_api_token,
    )


def main() -> None:
    """Run the local investigation API."""
    import uvicorn

    uvicorn.run(
        create_default_app(),
        host="127.0.0.1",
        port=8000,
    )
