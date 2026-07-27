import numpy as np
import polars as pl
import torch

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.graph.explain import extract_graph_edge_evidence
from aml_evidence_graph.graph.snapshots import (
    DailyGraphSnapshotBuilder,
    TemporalNodeIndexer,
    fit_edge_feature_scaler,
    graph_population_audit,
    transform_edge_features,
)
from aml_evidence_graph.models.graphsage import GraphSAGEEdgeClassifier
from aml_evidence_graph.training.graphsage import (
    GraphSAGETrainingConfig,
    fit_graphsage,
    predict_graphsage,
)


def test_daily_graph_snapshots_use_only_prior_days_and_score_edges() -> None:
    frame = pl.DataFrame(
        {
            CANONICAL.transaction_id: ["one", "two", "three"],
            CANONICAL.event_ts: pl.Series(
                [
                    "2022-10-07T10:00:00Z",
                    "2022-10-07T11:00:00Z",
                    "2022-10-08T10:00:00Z",
                ]
            ).str.to_datetime(time_zone="UTC"),
            CANONICAL.sender_account_id: ["a", "b", "b"],
            CANONICAL.receiver_account_id: ["b", "a", "a"],
            CANONICAL.source_row_number: [1, 2, 3],
            CANONICAL.is_laundering: [0, 1, 0],
            "amount": [10.0, 20.0, 30.0],
            "graph_sender_historical_out_degree": [0.0, 0.0, 1.0],
        }
    )
    indexer = TemporalNodeIndexer(unknown_hash_buckets=8).fit(frame.head(2))
    builder = DailyGraphSnapshotBuilder(
        indexer,
        edge_feature_columns=("amount", "graph_sender_historical_out_degree"),
    )

    snapshots = builder.build(frame)
    scaler = fit_edge_feature_scaler(snapshots[:1])
    scaled_snapshots = transform_edge_features(snapshots, scaler)
    second = scaled_snapshots[1]

    assert len(snapshots) == 2
    assert snapshots[0].history_edge_index.shape == (2, 0)
    assert snapshots[1].history_edge_index.shape == (2, 2)
    assert snapshots[1].transaction_ids == ("three",)
    assert np.isfinite(second.edge_features).all()

    model = GraphSAGEEdgeClassifier(
        num_nodes=indexer.num_nodes,
        edge_feature_dim=second.edge_features.shape[1],
        hidden_dim=8,
        num_layers=1,
    )
    logits = model(
        torch.arange(indexer.num_nodes),
        torch.from_numpy(second.history_edge_index),
        torch.from_numpy(second.scoring_edge_index),
        torch.from_numpy(second.edge_features),
    )

    assert logits.shape == (1,)
    evidence = extract_graph_edge_evidence(snapshots[1], scoring_edge_position=0)
    assert evidence.historical_source_out_degree == 1
    assert evidence.prior_directed_edge_count == 1


def test_scoring_snapshots_do_not_require_labels() -> None:
    frame = pl.DataFrame(
        {
            CANONICAL.transaction_id: ["one", "two"],
            CANONICAL.event_ts: pl.Series(
                [
                    "2023-07-01T10:00:00Z",
                    "2023-07-02T10:00:00Z",
                ]
            ).str.to_datetime(time_zone="UTC"),
            CANONICAL.sender_account_id: ["a", "b"],
            CANONICAL.receiver_account_id: ["b", "a"],
            CANONICAL.source_row_number: [1, 2],
            "amount": [10.0, 20.0],
        }
    )
    indexer = TemporalNodeIndexer(unknown_hash_buckets=8).fit(frame)

    snapshots = DailyGraphSnapshotBuilder(
        indexer,
        edge_feature_columns=("amount",),
    ).build(frame, include_labels=False)

    assert len(snapshots) == 2
    assert snapshots[0].labels.tolist() == [0]


def test_graph_population_audit_reports_overlap_without_identifiers() -> None:
    reference = pl.DataFrame(
        {
            CANONICAL.sender_account_id: ["a", "b"],
            CANONICAL.receiver_account_id: ["b", "c"],
        }
    )
    scored = pl.DataFrame(
        {
            CANONICAL.sender_account_id: ["a", "new"],
            CANONICAL.receiver_account_id: ["new", "c"],
        }
    )

    audit = graph_population_audit(reference, scored)

    assert audit["reference_account_count"] == 3
    assert audit["scored_account_count"] == 3
    assert audit["shared_account_count"] == 2
    assert audit["cold_start_account_count"] == 1
    assert audit["either_endpoint_cold_start_transaction_count"] == 2


def test_graphsage_training_uses_neighbor_loader_and_scores_all_validation_edges() -> None:
    frame = pl.DataFrame(
        {
            CANONICAL.transaction_id: [f"t{number}" for number in range(6)],
            CANONICAL.event_ts: pl.Series(
                [
                    "2022-10-07T10:00:00Z",
                    "2022-10-07T11:00:00Z",
                    "2022-10-08T10:00:00Z",
                    "2022-10-08T11:00:00Z",
                    "2023-05-01T10:00:00Z",
                    "2023-05-01T11:00:00Z",
                ]
            ).str.to_datetime(time_zone="UTC"),
            CANONICAL.sender_account_id: ["a", "b", "a", "c", "a", "c"],
            CANONICAL.receiver_account_id: ["b", "a", "c", "a", "c", "a"],
            CANONICAL.source_row_number: range(6),
            CANONICAL.is_laundering: [0, 1, 0, 1, 0, 1],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    indexer = TemporalNodeIndexer(unknown_hash_buckets=8).fit(frame.head(4))
    builder = DailyGraphSnapshotBuilder(
        indexer,
        edge_feature_columns=("amount",),
    )
    raw_train = builder.build(frame.head(4))
    raw_validation = builder.build(frame.tail(2))
    scaler = fit_edge_feature_scaler(raw_train)
    train_snapshots = transform_edge_features(raw_train, scaler)
    validation_snapshots = transform_edge_features(raw_validation, scaler)

    trained = fit_graphsage(
        train_snapshots,
        validation_snapshots,
        num_nodes=indexer.num_nodes,
        config=GraphSAGETrainingConfig(
            hidden_dim=8,
            num_layers=1,
            num_neighbors=(3,),
            batch_size=2,
            epochs=1,
            patience=1,
            device="cpu",
        ),
    )
    scores = predict_graphsage(
        trained,
        validation_snapshots,
        num_nodes=indexer.num_nodes,
    )

    assert len(scores) == 2
    assert np.isfinite(scores).all()
