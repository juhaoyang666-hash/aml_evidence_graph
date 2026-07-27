import polars as pl

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.training.graphsage import GraphSAGETrainingConfig
from aml_evidence_graph.training.oof import (
    generate_graphsage_oof_predictions,
    generate_table_oof_predictions,
    make_expanding_time_folds,
)


def _training_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for month in ("2022-10", "2022-11", "2022-12", "2023-01"):
        for day, label in ((1, 0), (2, 1), (3, 0), (4, 1)):
            rows.append(
                {
                    CANONICAL.transaction_id: f"{month}-{day}",
                    CANONICAL.event_ts: f"{month}-{day:02d}T12:00:00Z",
                    CANONICAL.is_laundering: label,
                    CANONICAL.sender_account_id: f"sender-{day % 2}",
                    CANONICAL.receiver_account_id: f"receiver-{day % 3}",
                    CANONICAL.source_row_number: len(rows) + 1,
                    CANONICAL.amount: float(day * 10),
                    "payment_type_feature": "A" if day % 2 else "B",
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col(CANONICAL.event_ts).str.to_datetime(time_zone="UTC")
    )


def test_expanding_time_oof_predictions_are_only_for_later_months() -> None:
    frame = _training_frame()
    folds = make_expanding_time_folds(frame, n_splits=2, minimum_training_months=2)
    predictions = generate_table_oof_predictions(
        frame,
        n_splits=2,
        minimum_training_months=2,
        catboost_params={"iterations": 10, "early_stopping_rounds": 3},
    )

    assert folds[0].training_months == ("2022-10", "2022-11")
    assert predictions.height == 8
    assert set(predictions["oof_fold_id"].to_list()) == {1, 2}
    catboost = predictions["catboost"]
    assert ((catboost >= 0) & (catboost <= 1)).all()


def test_graphsage_oof_predictions_do_not_use_later_month_edges() -> None:
    predictions = generate_graphsage_oof_predictions(
        _training_frame(),
        n_splits=2,
        minimum_training_months=2,
        config=GraphSAGETrainingConfig(
            hidden_dim=8,
            num_layers=1,
            num_neighbors=(3,),
            batch_size=4,
            epochs=1,
            patience=1,
            device="cpu",
        ),
    )

    assert predictions.height == 8
    graphsage = predictions["graphsage"]
    assert ((graphsage >= 0) & (graphsage <= 1)).all()
