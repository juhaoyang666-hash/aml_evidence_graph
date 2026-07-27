from pathlib import Path

import polars as pl
import pytest

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.training.fusion import (
    evaluate_persisted_fusion,
    fit_and_persist_fusion,
    merge_component_scores,
)


def _scores() -> pl.DataFrame:
    return pl.DataFrame(
        {
            CANONICAL.transaction_id: [f"t{index}" for index in range(8)],
            CANONICAL.is_laundering: [0, 1, 0, 1, 0, 1, 0, 1],
            "catboost": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.4, 0.6],
            "graphsage": [0.2, 0.9, 0.1, 0.8, 0.4, 0.7, 0.3, 0.6],
        }
    )


def test_fusion_runner_only_uses_oof_and_validation_inputs(tmp_path: Path) -> None:
    oof = _scores()
    validation = _scores()[::-1]
    summary = fit_and_persist_fusion(
        oof,
        validation,
        tmp_path / "fusion",
        component_models=("catboost", "graphsage"),
        alert_fraction=0.5,
    )

    assert summary.component_models == ["catboost", "graphsage"]
    assert (tmp_path / "fusion" / "oof_fusion.joblib").is_file()
    assert (tmp_path / "fusion" / "validation_calibration.joblib").is_file()
    assert (tmp_path / "fusion" / "run_manifest.json").is_file()

    evaluation = evaluate_persisted_fusion(
        tmp_path / "fusion",
        _scores(),
        tmp_path / "fusion-test",
    )

    assert evaluation.test_row_count == 8
    assert (tmp_path / "fusion-test" / "test_fusion_scores.parquet").is_file()
    assert (tmp_path / "fusion-test" / "run_manifest.json").is_file()


def test_component_scores_merge_only_on_equal_transaction_and_label_keys() -> None:
    first = _scores().select([CANONICAL.transaction_id, CANONICAL.is_laundering, "catboost"])
    second = _scores().select([CANONICAL.transaction_id, CANONICAL.is_laundering, "graphsage"])

    merged = merge_component_scores(
        [first, second],
        component_columns=["catboost", "graphsage"],
    )

    assert merged.height == first.height


def test_component_score_merge_rejects_nonmatching_coverage() -> None:
    first = _scores().select([CANONICAL.transaction_id, CANONICAL.is_laundering, "catboost"])
    second = _scores().slice(1).select(
        [CANONICAL.transaction_id, CANONICAL.is_laundering, "graphsage"]
    )

    with pytest.raises(ValueError, match="exactly the same"):
        merge_component_scores(
            [first, second],
            component_columns=["catboost", "graphsage"],
        )
