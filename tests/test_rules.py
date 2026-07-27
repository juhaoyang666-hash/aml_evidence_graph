from datetime import date

import polars as pl

from aml_evidence_graph.rules.engine import RuleDefinition, apply_rules


def test_only_approved_configured_rule_generates_feature_and_evidence() -> None:
    frame = pl.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "sender_cross_border_count_14d": [2.0, 3.0],
        }
    )
    rule = RuleDefinition(
        rule_id="R-1",
        version="test",
        status="approved",
        effective_from=date(2026, 1, 1),
        feature="sender_cross_border_count_14d",
        operator="gte",
        threshold=3.0,
        explanation_template="Threshold reached.",
        owner="test-owner",
        required_features=("sender_cross_border_count_14d",),
        approval_reference="approval-test",
        backtest_summary="Synthetic boundary test.",
    )

    features, hits = apply_rules(frame, [rule], as_of_date=date(2026, 1, 2))

    assert features["rule_R-1_hit"].to_list() == [0, 1]
    assert len(hits) == 1
    assert hits[0].transaction_id == "t2"
