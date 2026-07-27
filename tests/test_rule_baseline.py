import polars as pl

from aml_evidence_graph.training.table_baseline import rule_baseline_scores


def test_rule_baseline_uses_versioned_rule_hit_features_only() -> None:
    frame = pl.DataFrame(
        {
            "rule_R-A_hit": [0, 1, 1],
            "rule_R-B_hit": [0, 0, 1],
            "amount": [100.0, 200.0, 300.0],
        }
    )

    scores = rule_baseline_scores(frame)

    assert scores is not None
    assert scores.tolist() == [0.0, 1.0, 1.0]
