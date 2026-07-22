from pathlib import Path

import pandas as pd

from aml_evidence_graph.explain.tabular import write_catboost_explanations
from aml_evidence_graph.models.tabular import fit_table_models


def test_catboost_explanations_write_bounded_private_artifacts(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": [f"t{index}" for index in range(12)],
            "event_ts": pd.date_range("2022-10-07", periods=12, tz="UTC"),
            "source_row_number": range(12),
            "split": ["train"] * 6 + ["validation"] * 3 + ["test"] * 3,
            "is_laundering": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "laundering_type": ["None"] * 12,
            "amount_feature": [float(index) for index in range(12)],
            "payment_type_feature": ["A", "B"] * 6,
        }
    )
    models = fit_table_models(
        frame,
        catboost_params={"iterations": 10, "early_stopping_rounds": 3},
    )

    result = write_catboost_explanations(
        models,
        frame.loc[frame["split"] == "validation"],
        tmp_path,
        model_name="table",
        max_rows=2,
    )

    assert result["sample_count"] == 2
    assert Path(result["local_path"]).is_file()
    assert Path(result["global_path"]).is_file()
