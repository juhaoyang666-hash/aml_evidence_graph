import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from aml_evidence_graph.api.private_scoring import PrivateFeaturePartitionScoringService
from aml_evidence_graph.api.services import InMemoryEvidenceStore
from aml_evidence_graph.graph.snapshots import DailyGraphSnapshotBuilder, TemporalNodeIndexer
from aml_evidence_graph.models.graph_loading import load_graphsage_artifact
from aml_evidence_graph.models.graphsage import GraphSAGEEdgeClassifier
from aml_evidence_graph.models.loading import load_table_model_artifacts
from aml_evidence_graph.models.tabular import fit_table_models
from aml_evidence_graph.training.fusion import fit_and_persist_fusion
from aml_evidence_graph.training.graphsage import GraphSAGETrainingConfig


def test_persisted_table_models_can_be_loaded_for_private_scoring(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "transaction_id": [f"t{index}" for index in range(12)],
            "event_ts": [f"2022-10-{7 + index:02d}T00:00:00Z" for index in range(12)],
            "source_row_number": range(12),
            "split": ["train"] * 6 + ["validation"] * 3 + ["test"] * 3,
            "is_laundering": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "laundering_type": ["None"] * 12,
            "amount_feature": [float(index) for index in range(12)],
        }
    )
    models = fit_table_models(
        frame,
        catboost_params={"iterations": 10, "early_stopping_rounds": 3},
    )
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    import joblib

    joblib.dump(models.logistic, model_dir / "logistic.joblib")
    models.catboost.save_model(model_dir / "catboost.cbm")
    (model_dir / "feature_spec.json").write_text(
        '{"numeric_columns": ["amount_feature"], "categorical_columns": []}',
        encoding="utf-8",
    )

    loaded = load_table_model_artifacts(model_dir)

    assert len(loaded.predict_proba(frame.head(2))["catboost"]) == 2


def test_private_partition_scorer_uses_controlled_date_and_opaque_alerts(
    tmp_path: Path,
) -> None:
    frame = pl.DataFrame(
        {
            "transaction_id": [f"t{index}" for index in range(12)],
            "event_ts": [f"2022-10-{7 + index:02d}T00:00:00Z" for index in range(12)],
            "source_row_number": range(12),
            "split": ["train"] * 6 + ["validation"] * 3 + ["test"] * 3,
            "is_laundering": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "laundering_type": ["None"] * 12,
            "amount_feature": [float(index) for index in range(12)],
        }
    )
    models = fit_table_models(
        frame,
        catboost_params={"iterations": 10, "early_stopping_rounds": 3},
    )
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    import joblib

    joblib.dump(models.logistic, model_dir / "logistic.joblib")
    models.catboost.save_model(model_dir / "catboost.cbm")
    (model_dir / "feature_spec.json").write_text(
        '{"numeric_columns": ["amount_feature"], "categorical_columns": []}',
        encoding="utf-8",
    )
    feature_root = tmp_path / "features"
    target = feature_root / "event_date=2023-07-01" / "split=test"
    target.mkdir(parents=True)
    scoring_frame = frame.tail(3).drop("split")
    scoring_frame.write_parquet(target / "part.parquet")
    rule_evidence_dir = feature_root / "_rule_evidence"
    rule_evidence_dir.mkdir()
    (rule_evidence_dir / "event_date=2023-07-01.json").write_text(
        json.dumps(
            [
                {
                    "transaction_id": "t9",
                    "rule_id": "RULE-TEST",
                    "rule_version": "v1",
                    "feature": "amount_feature",
                    "observed_value": 9.0,
                    "threshold": 8.0,
                    "operator": "gte",
                    "explanation": "Synthetic test rule.",
                }
            ]
        ),
        encoding="utf-8",
    )
    store = InMemoryEvidenceStore()
    scorer = PrivateFeaturePartitionScoringService(
        feature_root=feature_root,
        table_model_dir=model_dir,
        evidence_store=store,
        alert_threshold=0,
        source_version="test-model",
    )

    result = scorer.score_partition("2023-07-01")

    assert len(result.alert_ids) == 3
    assert all(
        alert_id.startswith("alert-") and len(alert_id) == 38
        for alert_id in result.alert_ids
    )
    assert all(
        alert_id not in {"alert-t9", "alert-t10", "alert-t11"}
        for alert_id in result.alert_ids
    )
    evidence = store.get(result.alert_ids[0])
    assert evidence is not None
    assert evidence.rule_hits[0].rule_id == "RULE-TEST"


def test_persisted_graphsage_artifact_can_score_label_free_snapshots(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "transaction_id": ["one", "two"],
            "event_ts": pl.Series(
                ["2023-07-01T00:00:00Z", "2023-07-02T00:00:00Z"]
            ).str.to_datetime(time_zone="UTC"),
            "sender_account_id": ["a", "b"],
            "receiver_account_id": ["b", "a"],
            "source_row_number": [1, 2],
            "amount": [10.0, 20.0],
        }
    )
    indexer = TemporalNodeIndexer(unknown_hash_buckets=8).fit(frame)
    config = GraphSAGETrainingConfig(
        hidden_dim=8,
        num_layers=1,
        num_neighbors=(2,),
        batch_size=2,
        device="cpu",
    )
    model = GraphSAGEEdgeClassifier(
        num_nodes=indexer.num_nodes,
        edge_feature_dim=1,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    artifact_path = tmp_path / "graphsage.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "num_neighbors": config.num_neighbors,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "dropout": config.dropout,
                "patience": config.patience,
                "device": config.device,
                "random_seed": config.random_seed,
            },
            "edge_feature_columns": ("amount",),
            "scaler_mean": np.array([0.0]),
            "scaler_scale": np.array([1.0]),
            "known_node_index": indexer._known_nodes,
            "unknown_hash_buckets": indexer.unknown_hash_buckets,
        },
        artifact_path,
    )
    snapshots = DailyGraphSnapshotBuilder(
        indexer,
        edge_feature_columns=("amount",),
    ).build(frame, include_labels=False)

    loaded = load_graphsage_artifact(artifact_path, device="cpu")
    scores = loaded.predict(snapshots)

    assert len(scores) == 2
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_private_scorer_can_apply_frozen_graph_and_fusion_without_labels(
    tmp_path: Path,
) -> None:
    training = pl.DataFrame(
        {
            "transaction_id": [f"train-{index}" for index in range(12)],
            "event_ts": [f"2022-10-{7 + index:02d}T00:00:00Z" for index in range(12)],
            "source_row_number": range(12),
            "split": ["train"] * 6 + ["validation"] * 3 + ["test"] * 3,
            "is_laundering": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "laundering_type": ["None"] * 12,
            "amount": [float(index + 1) for index in range(12)],
        }
    )
    table_models = fit_table_models(
        training,
        catboost_params={"iterations": 10, "early_stopping_rounds": 3},
    )
    table_model_dir = tmp_path / "table_models" / "table_baselines"
    table_model_dir.mkdir(parents=True)
    import joblib

    joblib.dump(table_models.logistic, table_model_dir / "logistic.joblib")
    table_models.catboost.save_model(table_model_dir / "catboost.cbm")
    (table_model_dir / "feature_spec.json").write_text(
        '{"numeric_columns": ["amount"], "categorical_columns": []}',
        encoding="utf-8",
    )

    graph_training = pl.DataFrame(
        {
            "transaction_id": ["history-1", "current-1", "current-2"],
            "event_ts": pl.Series(
                [
                    "2023-07-01T00:00:00Z",
                    "2023-07-02T00:00:00Z",
                    "2023-07-02T01:00:00Z",
                ]
            ).str.to_datetime(time_zone="UTC"),
            "sender_account_id": ["a", "a", "b"],
            "receiver_account_id": ["b", "b", "a"],
            "source_row_number": [1, 2, 3],
            "amount": [1.0, 2.0, 3.0],
        }
    )
    indexer = TemporalNodeIndexer(unknown_hash_buckets=8).fit(graph_training.head(1))
    graph_config = GraphSAGETrainingConfig(
        hidden_dim=8,
        num_layers=1,
        num_neighbors=(2,),
        batch_size=2,
        device="cpu",
    )
    graph_model = GraphSAGEEdgeClassifier(
        num_nodes=indexer.num_nodes,
        edge_feature_dim=1,
        hidden_dim=graph_config.hidden_dim,
        num_layers=graph_config.num_layers,
        dropout=graph_config.dropout,
    )
    graph_path = tmp_path / "graphsage.pt"
    torch.save(
        {
            "state_dict": graph_model.state_dict(),
            "config": {
                "hidden_dim": graph_config.hidden_dim,
                "num_layers": graph_config.num_layers,
                "num_neighbors": graph_config.num_neighbors,
                "batch_size": graph_config.batch_size,
                "epochs": graph_config.epochs,
                "learning_rate": graph_config.learning_rate,
                "dropout": graph_config.dropout,
                "patience": graph_config.patience,
                "device": graph_config.device,
                "random_seed": graph_config.random_seed,
            },
            "edge_feature_columns": ("amount",),
            "scaler_mean": np.array([0.0]),
            "scaler_scale": np.array([1.0]),
            "known_node_index": indexer._known_nodes,
            "unknown_hash_buckets": indexer.unknown_hash_buckets,
        },
        graph_path,
    )

    fusion_scores = pl.DataFrame(
        {
            "transaction_id": [f"f{index}" for index in range(8)],
            "is_laundering": [0, 1, 0, 1, 0, 1, 0, 1],
            "catboost": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.4, 0.6],
            "graphsage": [0.2, 0.9, 0.1, 0.8, 0.4, 0.7, 0.3, 0.6],
        }
    )
    fusion_dir = tmp_path / "fusion"
    fit_and_persist_fusion(
        fusion_scores,
        fusion_scores[::-1],
        fusion_dir,
        component_models=("catboost", "graphsage"),
        alert_fraction=1,
    )

    feature_root = tmp_path / "features"
    history_target = feature_root / "event_date=2023-07-01" / "split=test"
    current_target = feature_root / "event_date=2023-07-02" / "split=test"
    history_target.mkdir(parents=True)
    current_target.mkdir(parents=True)
    graph_training.head(1).write_parquet(history_target / "part.parquet")
    graph_training.tail(2).write_parquet(current_target / "part.parquet")
    store = InMemoryEvidenceStore()
    scorer = PrivateFeaturePartitionScoringService(
        feature_root=feature_root,
        table_model_dir=table_model_dir,
        evidence_store=store,
        alert_threshold=0.5,
        source_version="frozen-fusion-test",
        selected_feature_names=("amount",),
        graphsage_artifact_path=graph_path,
        fusion_dir=fusion_dir,
        graphsage_device="cpu",
    )

    result = scorer.score_partition("2023-07-02")
    evidence = store.get(result.alert_ids[0])

    assert len(result.alert_ids) == 2
    assert evidence is not None
    assert evidence.fusion_probability is not None
    assert evidence.graph_evidence is not None
    assert "graphsage" in evidence.model_probabilities
