import polars as pl

from aml_evidence_graph.models.tabular import fit_table_models


def test_fit_table_models_excludes_labels_and_identifiers() -> None:
    frame = pl.DataFrame(
        {
            "transaction_id": [f"t{i}" for i in range(12)],
            "event_ts": [f"2022-10-{7 + i:02d}T00:00:00Z" for i in range(12)],
            "sender_account_id": [f"a{i}" for i in range(12)],
            "receiver_account_id": [f"b{i}" for i in range(12)],
            "source_row_number": range(12),
            "laundering_type": ["Normal"] * 12,
            "split": ["train"] * 6 + ["validation"] * 3 + ["test"] * 3,
            "is_laundering": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "amount_feature": [float(value) for value in range(12)],
            "payment_type_feature": ["A", "B"] * 6,
        }
    )

    models = fit_table_models(
        frame,
        catboost_params={"iterations": 10, "early_stopping_rounds": 3},
    )
    probabilities = models.predict_proba(frame.filter(pl.col("split") == "test"))

    assert models.feature_spec.all_columns == ("amount_feature", "payment_type_feature")
    assert len(probabilities["logistic"]) == 3
    assert len(probabilities["catboost"]) == 3
