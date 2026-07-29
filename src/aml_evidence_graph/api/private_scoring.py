"""Private local-file scoring with optional frozen graph and fusion artifacts."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pyarrow.dataset as ds

from aml_evidence_graph.api.services import EvidenceStore, ScoreBatchResult
from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.evidence.builder import build_risk_evidence_package
from aml_evidence_graph.graph.explain import (
    HistoricalGraphEvidenceIndex,
    build_historical_evidence_index,
    extract_graph_edge_evidence,
)
from aml_evidence_graph.graph.snapshots import DailyGraphSnapshotBuilder, TemporalGraphSnapshot
from aml_evidence_graph.models.loading import load_table_model_artifacts
from aml_evidence_graph.rules.engine import RuleHit
from aml_evidence_graph.training.fusion import load_persisted_fusion_artifacts

if TYPE_CHECKING:
    from aml_evidence_graph.models.graph_loading import LoadedGraphSAGEArtifact

GraphEvidenceContext = tuple[TemporalGraphSnapshot, int, HistoricalGraphEvidenceIndex]


@dataclass
class PrivateFeaturePartitionScoringService:
    """Score a configured private date partition; caller input never selects a file path."""

    feature_root: Path
    table_model_dir: Path
    evidence_store: EvidenceStore
    alert_threshold: float
    source_version: str
    selected_feature_names: tuple[str, ...] = (
        "sender_outgoing_count_7d",
        "receiver_incoming_count_7d",
        "graph_directed_edge_prior_count",
    )
    graphsage_artifact_path: Path | None = None
    fusion_dir: Path | None = None
    graphsage_device: str = "auto"
    _models: object = field(init=False, repr=False)
    _graphsage: LoadedGraphSAGEArtifact | None = field(init=False, default=None, repr=False)
    _graph_stat_models: object | None = field(init=False, default=None, repr=False)
    _fusion: object | None = field(init=False, default=None, repr=False)
    _calibration: object | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.alert_threshold <= 1:
            raise ValueError("alert_threshold must be in [0, 1].")
        self._models = load_table_model_artifacts(self.table_model_dir)
        if self.graphsage_artifact_path is not None:
            from aml_evidence_graph.models.graph_loading import load_graphsage_artifact

            self._graphsage = load_graphsage_artifact(
                self.graphsage_artifact_path,
                device=self.graphsage_device,
            )
        if self.fusion_dir is not None:
            self._fusion, self._calibration = load_persisted_fusion_artifacts(self.fusion_dir)
            component_names = set(self._fusion.model_names)
            supported = {"logistic", "catboost", "graph_stats_catboost", "graphsage"}
            unsupported = sorted(component_names.difference(supported))
            if unsupported:
                raise ValueError(
                    "Fusion uses unsupported score components: " + ", ".join(unsupported)
                )
            if "graphsage" in component_names and self._graphsage is None:
                raise ValueError(
                    "Fusion requires GraphSAGE but no graphsage artifact was configured."
                )
            if "graph_stats_catboost" in component_names:
                graph_stats_dir = self.table_model_dir.parent / "graph_stats_catboost"
                self._graph_stat_models = load_table_model_artifacts(graph_stats_dir)

    @staticmethod
    def _parse_event_date(partition_ref: str) -> date:
        try:
            return date.fromisoformat(partition_ref)
        except ValueError as error:
            raise ValueError(
                "partition_ref must be an ISO event date such as 2023-07-01."
            ) from error

    def _dataset(self) -> ds.Dataset:
        if not self.feature_root.is_dir():
            raise FileNotFoundError(f"Private feature root does not exist: {self.feature_root}")
        return ds.dataset(self.feature_root, format="parquet", partitioning="hive")

    @staticmethod
    def _read_partition(
        dataset: ds.Dataset,
        *,
        event_date: str,
        columns: set[str],
    ) -> pl.DataFrame:
        frame = pl.from_arrow(
            dataset.to_table(
                filter=ds.field("event_date") == event_date,
                columns=sorted(columns),
            )
        )
        if frame.is_empty():
            raise ValueError(f"No private feature rows for event date {event_date}.")
        return frame

    def _load_rule_hits(self, event_date: str) -> dict[str, list[RuleHit]]:
        evidence_path = self.feature_root / "_rule_evidence" / f"event_date={event_date}.json"
        if not evidence_path.is_file():
            return {}
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Rule evidence artifact must contain a list of rule hits.")
        result: dict[str, list[RuleHit]] = {}
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("Rule evidence artifact contains an invalid item.")
            hit = RuleHit(**item)
            result.setdefault(hit.transaction_id, []).append(hit)
        return result

    def _score_graphsage(
        self,
        dataset: ds.Dataset,
        *,
        event_day: date,
    ) -> tuple[dict[str, float], dict[str, GraphEvidenceContext], datetime] | None:
        if self._graphsage is None:
            return None
        event_date = event_day.isoformat()
        history_start = (
            event_day - timedelta(days=self._graphsage.config.history_window_days)
        ).isoformat()
        graph_columns = {
            CANONICAL.transaction_id,
            CANONICAL.event_ts,
            CANONICAL.sender_account_id,
            CANONICAL.receiver_account_id,
            CANONICAL.source_row_number,
            *self._graphsage.edge_feature_columns,
        }
        history = pl.from_arrow(
            dataset.to_table(
                filter=(ds.field("event_date") >= history_start)
                & (ds.field("event_date") < event_date),
                columns=sorted(graph_columns),
            )
        )
        current = self._read_partition(
            dataset,
            event_date=event_date,
            columns=graph_columns,
        )
        builder = DailyGraphSnapshotBuilder(
            self._graphsage.node_indexer,
            edge_feature_columns=self._graphsage.edge_feature_columns,
            history_window=timedelta(days=self._graphsage.config.history_window_days),
            store_relation_types=self._graphsage.config.architecture == "rgcn",
        )
        if not history.is_empty():
            builder.build(history, include_labels=False)
        current_snapshots = builder.build(current, include_labels=False)
        scores = self._graphsage.predict(current_snapshots)
        score_by_transaction: dict[str, float] = {}
        context_by_transaction: dict[str, GraphEvidenceContext] = {}
        offset = 0
        for snapshot in current_snapshots:
            history_index = build_historical_evidence_index(snapshot)
            for position, transaction_id in enumerate(snapshot.transaction_ids):
                score_by_transaction[transaction_id] = float(scores[offset + position])
                context_by_transaction[transaction_id] = (snapshot, position, history_index)
            offset += len(snapshot.transaction_ids)
        if offset != len(scores):
            raise AssertionError("GraphSAGE score count does not match scoring snapshots.")
        return (
            score_by_transaction,
            context_by_transaction,
            datetime.combine(event_day, datetime.min.time(), tzinfo=UTC),
        )

    def score_partition(self, partition_ref: str) -> ScoreBatchResult:
        """Score a controlled date partition using only preconfigured private artifacts."""
        event_day = self._parse_event_date(partition_ref)
        event_date = event_day.isoformat()
        dataset = self._dataset()
        required_table_columns = {
            CANONICAL.transaction_id,
            CANONICAL.event_ts,
            *self._models.feature_spec.all_columns,
        }
        available_columns = set(dataset.schema.names)
        missing_columns = sorted(required_table_columns.difference(available_columns))
        if missing_columns:
            raise ValueError(
                "Private feature dataset is missing required model columns: "
                + ", ".join(missing_columns)
            )
        table_columns = required_table_columns | (
            set(self.selected_feature_names).intersection(available_columns)
        )
        frame = self._read_partition(dataset, event_date=event_date, columns=table_columns)
        if frame[CANONICAL.transaction_id].is_duplicated().any():
            raise ValueError("Private feature partition has duplicate transaction IDs.")
        component_scores = self._models.predict_proba(frame)
        graph_result = self._score_graphsage(dataset, event_day=event_day)
        graph_context: dict[str, GraphEvidenceContext] = {}
        graph_snapshot_as_of: datetime | None = None
        if graph_result is not None:
            graph_scores, graph_context, graph_snapshot_as_of = graph_result
            component_scores["graphsage"] = np.asarray(
                [
                    graph_scores[str(transaction_id)]
                    for transaction_id in frame[CANONICAL.transaction_id]
                ],
                dtype=float,
            )
        if self._graph_stat_models is not None:
            component_scores["graph_stats_catboost"] = self._graph_stat_models.predict_proba(frame)[
                "catboost"
            ]

        fusion_scores: np.ndarray | None = None
        threshold = self.alert_threshold
        if self._fusion is not None:
            missing_components = sorted(
                set(self._fusion.model_names).difference(component_scores)
            )
            if missing_components:
                raise ValueError(
                    "Fusion components unavailable at scoring: "
                    + ", ".join(missing_components)
                )
            raw_fusion = self._fusion.predict_proba(
                pl.DataFrame(component_scores).select(self._fusion.model_names)
            )
            fusion_scores = self._calibration.predict_proba(raw_fusion)
            threshold = float(self._calibration.threshold)
        rule_hits_by_transaction = self._load_rule_hits(event_date)
        selected_features = tuple(
            feature for feature in self.selected_feature_names if feature in frame.columns
        )
        alert_ids: list[str] = []
        for row_position in range(frame.height):
            decision_score = (
                float(fusion_scores[row_position])
                if fusion_scores is not None
                else float(component_scores["catboost"][row_position])
            )
            if decision_score < threshold:
                continue
            transaction = frame.row(row_position, named=True)
            transaction_id = str(transaction[CANONICAL.transaction_id])
            alert_id = f"alert-{uuid.uuid4().hex}"
            source_versions = {"table_model": self.source_version}
            if self._graphsage is not None:
                source_versions["graphsage_model"] = self.source_version
            if self._fusion is not None:
                source_versions["fusion_and_calibration"] = self.source_version
            missing_evidence = []
            if graph_result is None:
                missing_evidence.append(
                    "Batch scoring did not attach a GraphSAGE local path explanation."
                )
            context = graph_context.get(transaction_id)
            graph_edge_evidence = (
                extract_graph_edge_evidence(
                    context[0],
                    scoring_edge_position=context[1],
                    history_index=context[2],
                )
                if context is not None
                else None
            )
            evidence = build_risk_evidence_package(
                transaction,
                alert_id=alert_id,
                model_probabilities={
                    name: float(values[row_position])
                    for name, values in component_scores.items()
                },
                fusion_probability=(
                    float(fusion_scores[row_position]) if fusion_scores is not None else None
                ),
                source_versions=source_versions,
                selected_feature_names=selected_features,
                rule_hits=rule_hits_by_transaction.get(transaction_id, []),
                graph_edge_evidence=graph_edge_evidence,
                graph_snapshot_as_of=graph_snapshot_as_of,
                missing_evidence=missing_evidence,
                uncertainty_notes=[
                    "A risk probability is an alert priority, not a case decision."
                ],
            )
            self.evidence_store.put(evidence)
            alert_ids.append(alert_id)
        return ScoreBatchResult(
            partition_ref=event_date,
            alert_ids=alert_ids,
            model_version=self.source_version,
        )
