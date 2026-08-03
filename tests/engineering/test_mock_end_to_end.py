from pathlib import Path

import polars as pl
import yaml

from aml_evidence_graph.features.build import build_pit_feature_dataset
from aml_evidence_graph.ingestion.prepare import convert_csv_to_parquet
from aml_evidence_graph.training.graphsage import GraphSAGETrainingConfig
from aml_evidence_graph.training.run_graphsage import train_and_evaluate_graphsage
from aml_evidence_graph.training.table_baseline import train_and_evaluate_table_baselines

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mock_container_configuration_excludes_private_data_and_secrets() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignored = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.demo.yml").read_text(encoding="utf-8")
    )

    assert "COPY configs ./configs" in dockerfile
    assert "COPY knowledge ./knowledge" in dockerfile
    assert "FROM python:3.11-slim" in dockerfile
    assert "COPY data" not in dockerfile
    assert "COPY artifacts" not in dockerfile
    assert "data/" in ignored
    assert "artifacts/" in ignored
    assert "volumes" not in compose["services"]["aml-demo"]
    assert compose["services"]["aml-demo"]["environment"]["AML_LLM_ENABLED"] == "false"


def _mock_raw_transactions() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    split_dates = (
        ("2022-10-07", 6),
        ("2023-05-01", 6),
        ("2023-07-01", 6),
    )
    row_number = 0
    for event_date, count in split_dates:
        for offset in range(count):
            row_number += 1
            rows.append(
                {
                    "Time": f"12:00:{offset:02d}",
                    "Date": event_date,
                    "Sender_account": f"mock_sender_{row_number % 4}",
                    "Receiver_account": f"mock_receiver_{row_number % 5}",
                    "Amount": float(10 + row_number),
                    "Payment_currency": "USD",
                    "Received_currency": "USD",
                    "Sender_bank_location": "US",
                    "Receiver_bank_location": "CA" if offset % 2 else "US",
                    "Payment_type": "CASH" if offset % 2 else "TRANSFER",
                    "Is_laundering": 1 if offset in {1, 4} else 0,
                    "Laundering_type": "Mock type" if offset in {1, 4} else "None",
                }
            )
    return pl.DataFrame(rows)


def test_mock_csv_to_features_to_table_metrics(tmp_path: Path) -> None:
    raw_csv = tmp_path / "mock.csv"
    _mock_raw_transactions().write_csv(raw_csv)
    prepared_root = tmp_path / "prepared"
    feature_root = tmp_path / "features"
    model_root = tmp_path / "models"
    graph_model_root = tmp_path / "graph-models"

    conversion_summary = convert_csv_to_parquet(
        raw_csv,
        prepared_root,
        chunk_size=5,
    )
    build_pit_feature_dataset(prepared_root, feature_root)
    summary = train_and_evaluate_table_baselines(
        feature_root,
        model_root,
        maximum_training_negative_rows=3,
        catboost_params={"iterations": 10, "early_stopping_rounds": 3},
    )
    graph_summary = train_and_evaluate_graphsage(
        feature_root,
        graph_model_root,
        config=GraphSAGETrainingConfig(
            hidden_dim=8,
            num_layers=1,
            num_neighbors=(3,),
            batch_size=3,
            epochs=1,
            patience=1,
            device="cpu",
        ),
    )

    assert summary.training_rows_before_sampling == 6
    assert conversion_summary.row_count == 18
    assert (prepared_root / "_run_manifest.json").is_file()
    assert summary.training_rows_after_sampling == 5
    assert summary.validation_rows == 6
    assert summary.test_rows == 6
    assert set(summary.test_metrics) == {"logistic", "catboost", "graph_stats_catboost"}
    assert (model_root / "metrics.json").is_file()
    assert (model_root / "run_manifest.json").is_file()
    assert summary.test_monthly_stability
    assert graph_summary.test_snapshot_count == 1
    assert (graph_model_root / "graphsage.pt").is_file()
    assert graph_summary.test_metrics["sample_count"] == 6
