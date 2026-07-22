import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from aml_evidence_graph.aggregation.views import (
    _join_private_scores,
    build_investigation_views,
    main,
    write_investigation_views,
)
from aml_evidence_graph.data.contract import CANONICAL


def _scored_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            CANONICAL.transaction_id: ["t1", "t-same", "t2", "t3", "t-low"],
            CANONICAL.event_ts: [
                "2023-07-01T10:00:00Z",
                "2023-07-01T10:00:00Z",
                "2023-07-01T11:00:00Z",
                "2023-07-01T12:00:00Z",
                "2023-07-01T11:00:00Z",
            ],
            CANONICAL.sender_account_id: ["acct-a", "acct-b", "acct-b", "acct-c", "acct-a"],
            CANONICAL.receiver_account_id: ["acct-b", "acct-c", "acct-c", "acct-d", "acct-e"],
            CANONICAL.amount: [10.0, 15.0, 20.0, 30.0, 5.0],
            CANONICAL.source_row_number: [1, 2, 3, 4, 5],
            "risk_score": [0.90, 0.95, 0.80, 0.70, 0.20],
            "rule_hit_count": [1, 0, 1, 0, 0],
        }
    )


def test_investigation_views_aggregate_scores_without_account_labels(tmp_path: Path) -> None:
    views = build_investigation_views(
        _scored_transactions(),
        as_of_ts="2023-07-01T12:00:00Z",
        score_column="risk_score",
        window_days=1,
        risk_threshold=0.6,
        top_n=2,
    )

    account_a = views.account_risk.set_index("account_id").loc["acct-a"]
    assert account_a["scored_transaction_count"] == 2
    assert account_a["alert_transaction_count"] == 1
    assert account_a["max_risk_score"] == 0.90
    assert account_a["top_n_mean_risk_score"] == 0.55
    assert len(views.case_views) == 1
    assert views.case_views[0].state == "investigation_candidate"
    assert views.case_views[0].transaction_count == 4
    assert views.case_views[0].funds_path_count == len(views.funds_paths)
    assert all(
        earlier < later
        for path in views.funds_paths
        for earlier, later in zip(
            path.event_timestamps,
            path.event_timestamps[1:],
            strict=False,
        )
    )
    assert not any(path.transaction_ids == ("t1", "t-same") for path in views.funds_paths)

    summary = write_investigation_views(tmp_path / "views", views)
    assert summary["account_count"] == 5
    assert (tmp_path / "views" / "account_risk.parquet").is_file()
    assert (tmp_path / "views" / "funds_paths.json").is_file()
    assert (tmp_path / "views" / "case_views.json").is_file()


def test_investigation_views_reject_labels_and_future_events() -> None:
    labelled = _scored_transactions().assign(**{CANONICAL.is_laundering: [0, 1, 0, 0, 0]})
    with pytest.raises(ValueError, match="must not receive labels"):
        build_investigation_views(labelled, as_of_ts="2023-07-01T12:00:00Z")

    with pytest.raises(ValueError, match="after the explicit as_of_ts"):
        build_investigation_views(
            _scored_transactions(),
            as_of_ts="2023-07-01T11:00:00Z",
        )


def test_private_score_join_uses_explicit_id_equality_and_drops_labels() -> None:
    transactions = _scored_transactions().assign(
        **{
            CANONICAL.is_laundering: [0, 1, 0, 0, 0],
            CANONICAL.laundering_type: ["", "x", "", "", ""],
            "rule_demo_hit": [1, 0, 1, 0, 0],
        }
    )
    scores = transactions.loc[:, [CANONICAL.transaction_id, "risk_score"]].copy()
    joined = _join_private_scores(transactions, scores, score_column="risk_score")

    assert CANONICAL.is_laundering not in joined
    assert CANONICAL.laundering_type not in joined
    assert joined["rule_hit_count"].tolist() == [1, 0, 1, 0, 0]

    with pytest.raises(ValueError, match="absent from the transaction"):
        _join_private_scores(
            transactions,
            pd.concat(
                [
                    scores.iloc[:-1],
                    pd.DataFrame({CANONICAL.transaction_id: ["unknown"], "risk_score": [0.9]}),
                ],
                ignore_index=True,
            ),
            score_column="risk_score",
        )


def test_investigation_view_cli_writes_private_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transactions = _scored_transactions().assign(
        **{CANONICAL.is_laundering: [0, 1, 0, 0, 0]}
    )
    transaction_path = tmp_path / "transactions.parquet"
    score_path = tmp_path / "scores.parquet"
    output_dir = tmp_path / "views"
    transactions.to_parquet(transaction_path, index=False)
    transactions.loc[:, [CANONICAL.transaction_id, "risk_score"]].to_parquet(
        score_path,
        index=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aml-build-investigation-views",
            "--transactions",
            str(transaction_path),
            "--scores",
            str(score_path),
            "--score-column",
            "risk_score",
            "--as-of-ts",
            "2023-07-01T12:00:00Z",
            "--output",
            str(output_dir),
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["account_count"] == 5
    assert (output_dir / "run_manifest.json").is_file()
    assert (output_dir / "case_views.json").is_file()
