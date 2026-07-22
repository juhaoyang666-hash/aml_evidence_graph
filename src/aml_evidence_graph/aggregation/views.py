"""Create bounded post-score account, path, and investigation-case views.

These views are an investigation aid, not another supervised target and not a
transaction-scoring feature.  They deliberately reject transaction labels and
only derive their values from already-produced risk scores and rule evidence.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import heapq
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.tracking.run import create_run_manifest

LABEL_COLUMNS = frozenset({CANONICAL.is_laundering, CANONICAL.laundering_type})


@dataclass
class _AccountAccumulator:
    """Bounded score summary for one tokenized account."""

    top_n: int
    scored_transaction_count: int = 0
    alert_transaction_count: int = 0
    sender_transaction_count: int = 0
    receiver_transaction_count: int = 0
    rule_hit_count: int = 0
    max_risk_score: float = 0.0
    _top_scores: list[float] = field(default_factory=list)

    def observe(
        self,
        *,
        score: float,
        is_alert: bool,
        rule_hit_count: int,
        role: str,
    ) -> None:
        self.scored_transaction_count += 1
        self.alert_transaction_count += int(is_alert)
        self.rule_hit_count += rule_hit_count
        if role in {"sender", "both"}:
            self.sender_transaction_count += 1
        if role in {"receiver", "both"}:
            self.receiver_transaction_count += 1
        self.max_risk_score = max(self.max_risk_score, score)
        if len(self._top_scores) < self.top_n:
            heapq.heappush(self._top_scores, score)
        elif score > self._top_scores[0]:
            heapq.heapreplace(self._top_scores, score)

    @property
    def top_n_mean_risk_score(self) -> float:
        return float(sum(self._top_scores) / len(self._top_scores)) if self._top_scores else 0.0


@dataclass(frozen=True)
class FundsPath:
    """A bounded, strictly time-forward chain of high-risk transaction edges."""

    path_id: str
    transaction_ids: tuple[str, ...]
    account_ids: tuple[str, ...]
    event_timestamps: tuple[str, ...]
    risk_scores: tuple[float, ...]
    total_amount: float


@dataclass(frozen=True)
class InvestigationCaseView:
    """A connected high-risk subgraph candidate requiring human investigation."""

    case_id: str
    state: str
    transaction_ids: tuple[str, ...]
    account_ids: tuple[str, ...]
    first_event_ts: str
    last_event_ts: str
    transaction_count: int
    account_count: int
    max_risk_score: float
    top_n_mean_risk_score: float
    rule_hit_count: int
    funds_path_count: int


@dataclass(frozen=True)
class InvestigationViews:
    """All bounded views for one explicit as-of time and historical window."""

    as_of_ts: str
    window_start_ts: str
    account_risk: pd.DataFrame
    funds_paths: tuple[FundsPath, ...]
    case_views: tuple[InvestigationCaseView, ...]


@dataclass(frozen=True)
class _ScoredEdge:
    transaction_id: str
    event_ts: pd.Timestamp
    sender_account_id: str
    receiver_account_id: str
    amount: float
    risk_score: float
    source_row_number: int
    rule_hit_count: int


def _as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="raise")
    if pd.isna(timestamp):
        raise ValueError("Timestamp cannot be null.")
    return pd.Timestamp(timestamp)


def _as_iso_timestamp(value: pd.Timestamp) -> str:
    return value.to_pydatetime().isoformat()


def _stable_identifier(prefix: str, values: tuple[str, ...]) -> str:
    digest = hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _normalise_scored_transactions(
    scored_transactions: pd.DataFrame,
    *,
    score_column: str,
    as_of_ts: str | pd.Timestamp,
    window_days: int,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    if window_days < 1:
        raise ValueError("window_days must be positive.")
    forbidden = sorted(LABEL_COLUMNS.intersection(scored_transactions.columns))
    if forbidden:
        raise ValueError(
            "Post-score investigation views must not receive labels: " + ", ".join(forbidden)
        )
    required = {
        CANONICAL.transaction_id,
        CANONICAL.event_ts,
        CANONICAL.sender_account_id,
        CANONICAL.receiver_account_id,
        CANONICAL.amount,
        CANONICAL.source_row_number,
        score_column,
    }
    missing = sorted(required.difference(scored_transactions.columns))
    if missing:
        raise ValueError("Scored transactions are missing: " + ", ".join(missing))

    as_of = _as_utc_timestamp(as_of_ts)
    window_start = as_of - pd.Timedelta(days=window_days)
    columns = list(required)
    if "rule_hit_count" in scored_transactions.columns:
        columns.append("rule_hit_count")
    frame = scored_transactions.loc[:, columns].copy()
    frame[CANONICAL.event_ts] = pd.to_datetime(
        frame[CANONICAL.event_ts], utc=True, errors="raise"
    )
    if frame[CANONICAL.event_ts].isna().any():
        raise ValueError("event_ts cannot contain null timestamps.")
    if (frame[CANONICAL.event_ts] > as_of).any():
        raise ValueError("Scored transactions contain events after the explicit as_of_ts.")
    if frame[CANONICAL.transaction_id].isna().any() or frame[
        CANONICAL.transaction_id
    ].duplicated().any():
        raise ValueError("transaction_id must be non-null and unique.")
    for account_column in (CANONICAL.sender_account_id, CANONICAL.receiver_account_id):
        is_missing_or_empty = frame[account_column].isna().any() or (
            frame[account_column].astype(str).str.len().eq(0).any()
        )
        if is_missing_or_empty:
            raise ValueError(f"{account_column} must contain non-empty account tokens.")
        frame[account_column] = frame[account_column].astype(str)
    frame[CANONICAL.transaction_id] = frame[CANONICAL.transaction_id].astype(str)
    frame[CANONICAL.amount] = pd.to_numeric(frame[CANONICAL.amount], errors="raise")
    if frame[CANONICAL.amount].isna().any() or (frame[CANONICAL.amount] < 0).any():
        raise ValueError("amount must be non-null and non-negative.")
    frame[score_column] = pd.to_numeric(frame[score_column], errors="raise")
    if frame[score_column].isna().any() or not frame[score_column].between(0, 1).all():
        raise ValueError(f"{score_column} must contain probabilities in [0, 1].")
    source_rows = pd.to_numeric(frame[CANONICAL.source_row_number], errors="raise")
    if source_rows.isna().any() or (source_rows < 1).any():
        raise ValueError("source_row_number must be positive.")
    frame[CANONICAL.source_row_number] = source_rows.astype(int)
    if "rule_hit_count" not in frame:
        frame["rule_hit_count"] = 0
    else:
        rule_hits = pd.to_numeric(frame["rule_hit_count"], errors="raise")
        if rule_hits.isna().any() or (rule_hits < 0).any() or not (rule_hits % 1 == 0).all():
            raise ValueError("rule_hit_count must be a non-negative integer.")
        frame["rule_hit_count"] = rule_hits.astype(int)
    frame = frame.loc[
        frame[CANONICAL.event_ts].ge(window_start)
        & frame[CANONICAL.event_ts].le(as_of)
    ].copy()
    return frame, as_of, window_start


def _build_account_risk(
    frame: pd.DataFrame,
    *,
    score_column: str,
    risk_threshold: float,
    top_n: int,
    as_of: pd.Timestamp,
    window_start: pd.Timestamp,
) -> pd.DataFrame:
    accumulators: dict[str, _AccountAccumulator] = {}
    columns = [
        CANONICAL.sender_account_id,
        CANONICAL.receiver_account_id,
        score_column,
        "rule_hit_count",
    ]
    for sender, receiver, score, rule_hits in frame.loc[:, columns].itertuples(
        index=False, name=None
    ):
        score_value = float(score)
        alert = score_value >= risk_threshold
        accounts = ((str(sender), "both"),) if sender == receiver else (
            (str(sender), "sender"),
            (str(receiver), "receiver"),
        )
        for account_id, role in accounts:
            accumulator = accumulators.setdefault(account_id, _AccountAccumulator(top_n=top_n))
            accumulator.observe(
                score=score_value,
                is_alert=alert,
                rule_hit_count=int(rule_hits),
                role=role,
            )
    records = [
        {
            "account_id": account_id,
            "as_of_ts": _as_iso_timestamp(as_of),
            "window_start_ts": _as_iso_timestamp(window_start),
            "scored_transaction_count": value.scored_transaction_count,
            "alert_transaction_count": value.alert_transaction_count,
            "sender_transaction_count": value.sender_transaction_count,
            "receiver_transaction_count": value.receiver_transaction_count,
            "max_risk_score": value.max_risk_score,
            "top_n_mean_risk_score": value.top_n_mean_risk_score,
            "rule_hit_count": value.rule_hit_count,
        }
        for account_id, value in accumulators.items()
    ]
    columns_out = [
        "account_id",
        "as_of_ts",
        "window_start_ts",
        "scored_transaction_count",
        "alert_transaction_count",
        "sender_transaction_count",
        "receiver_transaction_count",
        "max_risk_score",
        "top_n_mean_risk_score",
        "rule_hit_count",
    ]
    return pd.DataFrame(records, columns=columns_out).sort_values(
        ["max_risk_score", "account_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def _high_risk_edges(
    frame: pd.DataFrame,
    *,
    score_column: str,
    risk_threshold: float,
) -> list[_ScoredEdge]:
    high_risk = frame.loc[frame[score_column].ge(risk_threshold)].copy()
    high_risk = high_risk.sort_values(
        [CANONICAL.event_ts, CANONICAL.source_row_number, CANONICAL.transaction_id], kind="stable"
    )
    return [
        _ScoredEdge(
            transaction_id=str(transaction_id),
            event_ts=pd.Timestamp(event_ts),
            sender_account_id=str(sender),
            receiver_account_id=str(receiver),
            amount=float(amount),
            risk_score=float(score),
            source_row_number=int(source_row),
            rule_hit_count=int(rule_hits),
        )
        for transaction_id, event_ts, sender, receiver, amount, score, source_row, rule_hits in (
            high_risk.loc[
                :,
                [
                    CANONICAL.transaction_id,
                    CANONICAL.event_ts,
                    CANONICAL.sender_account_id,
                    CANONICAL.receiver_account_id,
                    CANONICAL.amount,
                    score_column,
                    CANONICAL.source_row_number,
                    "rule_hit_count",
                ],
            ].itertuples(index=False, name=None)
        )
    ]


def _path_from_indices(edges: list[_ScoredEdge], indices: tuple[int, ...]) -> FundsPath:
    selected = [edges[index] for index in indices]
    transaction_ids = tuple(edge.transaction_id for edge in selected)
    account_ids = (selected[0].sender_account_id,) + tuple(
        edge.receiver_account_id for edge in selected
    )
    return FundsPath(
        path_id=_stable_identifier("funds", transaction_ids),
        transaction_ids=transaction_ids,
        account_ids=account_ids,
        event_timestamps=tuple(_as_iso_timestamp(edge.event_ts) for edge in selected),
        risk_scores=tuple(edge.risk_score for edge in selected),
        total_amount=float(sum(edge.amount for edge in selected)),
    )


def _build_funds_paths(
    edges: list[_ScoredEdge],
    *,
    max_hops: int,
    max_paths: int,
    max_branching: int,
) -> tuple[FundsPath, ...]:
    if max_hops < 2:
        raise ValueError("max_hops must be at least 2 for a funds path.")
    if max_paths < 1 or max_branching < 1:
        raise ValueError("max_paths and max_branching must be positive.")
    outgoing: dict[str, list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        outgoing[edge.sender_account_id].append(index)
    times_by_account = {
        account_id: [edges[index].event_ts for index in indices]
        for account_id, indices in outgoing.items()
    }
    paths: list[FundsPath] = []
    seen_paths: set[str] = set()

    def extend(indices: tuple[int, ...], visited_accounts: frozenset[str]) -> None:
        if len(paths) >= max_paths:
            return
        current = edges[indices[-1]]
        if len(indices) >= 2:
            path = _path_from_indices(edges, indices)
            if path.path_id not in seen_paths:
                seen_paths.add(path.path_id)
                paths.append(path)
            if len(paths) >= max_paths:
                return
        if len(indices) == max_hops:
            return
        candidate_indices = outgoing.get(current.receiver_account_id, [])
        candidate_times = times_by_account.get(current.receiver_account_id, [])
        start = bisect.bisect_right(candidate_times, current.event_ts)
        ordered_candidates = sorted(
            candidate_indices[start:],
            key=lambda index: (
                -edges[index].risk_score,
                edges[index].event_ts,
                edges[index].source_row_number,
                edges[index].transaction_id,
            ),
        )[:max_branching]
        for next_index in ordered_candidates:
            candidate = edges[next_index]
            if candidate.receiver_account_id in visited_accounts:
                continue
            extend(
                indices + (next_index,),
                visited_accounts | {candidate.receiver_account_id},
            )
            if len(paths) >= max_paths:
                return

    for root_index, root in enumerate(edges):
        extend(
            (root_index,),
            frozenset({root.sender_account_id, root.receiver_account_id}),
        )
        if len(paths) >= max_paths:
            break
    return tuple(paths)


def _build_case_views(
    edges: list[_ScoredEdge],
    paths: tuple[FundsPath, ...],
    *,
    top_n: int,
) -> tuple[InvestigationCaseView, ...]:
    if not edges:
        return ()
    parent = list(range(len(edges)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    account_owner: dict[str, int] = {}
    for index, edge in enumerate(edges):
        for account_id in {edge.sender_account_id, edge.receiver_account_id}:
            owner = account_owner.setdefault(account_id, index)
            union(index, owner)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(edges)):
        components[find(index)].append(index)
    ordered_components = sorted(
        components.values(),
        key=lambda indices: (
            edges[min(indices)].event_ts,
            edges[min(indices)].source_row_number,
            edges[min(indices)].transaction_id,
        ),
    )
    component_by_transaction = {
        edges[index].transaction_id: root
        for root, indices in components.items()
        for index in indices
    }
    path_count_by_component = defaultdict(int)
    for path in paths:
        path_components = {
            component_by_transaction[transaction_id]
            for transaction_id in path.transaction_ids
        }
        if len(path_components) != 1:
            raise AssertionError("A directed funds path must remain inside one case component.")
        path_count_by_component[path_components.pop()] += 1

    result: list[InvestigationCaseView] = []
    for indices in ordered_components:
        selected = [edges[index] for index in indices]
        selected.sort(key=lambda edge: (edge.event_ts, edge.source_row_number, edge.transaction_id))
        transaction_ids = tuple(edge.transaction_id for edge in selected)
        account_ids = tuple(
            sorted(
                {
                    account_id
                    for edge in selected
                    for account_id in (edge.sender_account_id, edge.receiver_account_id)
                }
            )
        )
        top_scores = sorted((edge.risk_score for edge in selected), reverse=True)[:top_n]
        result.append(
            InvestigationCaseView(
                case_id=_stable_identifier("case", transaction_ids),
                state="investigation_candidate",
                transaction_ids=transaction_ids,
                account_ids=account_ids,
                first_event_ts=_as_iso_timestamp(selected[0].event_ts),
                last_event_ts=_as_iso_timestamp(selected[-1].event_ts),
                transaction_count=len(selected),
                account_count=len(account_ids),
                max_risk_score=max(edge.risk_score for edge in selected),
                top_n_mean_risk_score=float(sum(top_scores) / len(top_scores)),
                rule_hit_count=sum(edge.rule_hit_count for edge in selected),
                funds_path_count=path_count_by_component[find(indices[0])],
            )
        )
    return tuple(result)


def build_investigation_views(
    scored_transactions: pd.DataFrame,
    *,
    as_of_ts: str | pd.Timestamp,
    score_column: str = "risk_score",
    window_days: int = 30,
    risk_threshold: float = 0.5,
    top_n: int = 3,
    max_hops: int = 3,
    max_paths: int = 1_000,
    max_branching: int = 5,
    max_high_risk_transactions: int = 100_000,
) -> InvestigationViews:
    """Aggregate only post-score evidence into account, path, and case views.

    ``as_of_ts`` is mandatory so an operator cannot silently aggregate future
    scores.  The resulting object is for human investigation only; it must not
    be joined back into the transaction risk model.
    """
    if not 0 <= risk_threshold <= 1:
        raise ValueError("risk_threshold must be in [0, 1].")
    if top_n < 1 or max_high_risk_transactions < 1:
        raise ValueError("top_n and max_high_risk_transactions must be positive.")
    frame, as_of, window_start = _normalise_scored_transactions(
        scored_transactions,
        score_column=score_column,
        as_of_ts=as_of_ts,
        window_days=window_days,
    )
    account_risk = _build_account_risk(
        frame,
        score_column=score_column,
        risk_threshold=risk_threshold,
        top_n=top_n,
        as_of=as_of,
        window_start=window_start,
    )
    edges = _high_risk_edges(frame, score_column=score_column, risk_threshold=risk_threshold)
    if len(edges) > max_high_risk_transactions:
        raise ValueError(
            "High-risk transaction count exceeds max_high_risk_transactions; "
            "raise the threshold or explicitly increase the bounded limit."
        )
    paths = _build_funds_paths(
        edges,
        max_hops=max_hops,
        max_paths=max_paths,
        max_branching=max_branching,
    )
    cases = _build_case_views(edges, paths, top_n=top_n)
    return InvestigationViews(
        as_of_ts=_as_iso_timestamp(as_of),
        window_start_ts=_as_iso_timestamp(window_start),
        account_risk=account_risk,
        funds_paths=paths,
        case_views=cases,
    )


def write_investigation_views(output_dir: Path, views: InvestigationViews) -> dict[str, Any]:
    """Write private view artifacts; summary metadata intentionally has no IDs."""
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    views.account_risk.to_parquet(output_dir / "account_risk.parquet", index=False)
    (output_dir / "funds_paths.json").write_text(
        json.dumps([asdict(path) for path in views.funds_paths], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "case_views.json").write_text(
        json.dumps([asdict(case) for case in views.case_views], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "as_of_ts": views.as_of_ts,
        "window_start_ts": views.window_start_ts,
        "account_count": len(views.account_risk),
        "funds_path_count": len(views.funds_paths),
        "investigation_case_count": len(views.case_views),
        "privacy_notice": (
            "Account tokens, transaction tokens, and detailed views are private artifacts. "
            "This summary contains aggregate counts only."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _read_private_parquet(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    if path.is_dir():
        return ds.dataset(
            path,
            format="parquet",
            partitioning="hive",
            exclude_invalid_files=True,
        ).to_table().to_pandas()
    raise FileNotFoundError(f"Private input does not exist: {path}")


def _join_private_scores(
    transactions: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    score_column: str,
) -> pd.DataFrame:
    transaction_columns = [
        CANONICAL.transaction_id,
        CANONICAL.event_ts,
        CANONICAL.sender_account_id,
        CANONICAL.receiver_account_id,
        CANONICAL.amount,
        CANONICAL.source_row_number,
    ]
    missing_transactions = sorted(set(transaction_columns).difference(transactions.columns))
    if missing_transactions:
        raise ValueError("Transaction input is missing: " + ", ".join(missing_transactions))
    missing_scores = sorted({CANONICAL.transaction_id, score_column}.difference(scores.columns))
    if missing_scores:
        raise ValueError("Score input is missing: " + ", ".join(missing_scores))
    if transactions[CANONICAL.transaction_id].duplicated().any() or scores[
        CANONICAL.transaction_id
    ].duplicated().any():
        raise ValueError("Transaction and score inputs require unique transaction IDs.")
    transaction_ids = set(transactions[CANONICAL.transaction_id].astype(str))
    score_ids = set(scores[CANONICAL.transaction_id].astype(str))
    unknown_score_ids = score_ids.difference(transaction_ids)
    if unknown_score_ids:
        raise ValueError(
            "Score input contains transaction IDs absent from the transaction input."
        )
    rule_columns = sorted(
        column
        for column in transactions.columns
        if column.startswith("rule_") and column.endswith("_hit")
    )
    left = transactions.loc[
        transactions[CANONICAL.transaction_id].astype(str).isin(score_ids),
        [*transaction_columns, *rule_columns],
    ].copy()
    right = scores.loc[:, [CANONICAL.transaction_id, score_column]].copy()
    left[CANONICAL.transaction_id] = left[CANONICAL.transaction_id].astype(str)
    right[CANONICAL.transaction_id] = right[CANONICAL.transaction_id].astype(str)
    joined = left.merge(
        right,
        on=CANONICAL.transaction_id,
        how="inner",
        validate="one_to_one",
    )
    if rule_columns:
        joined["rule_hit_count"] = joined.loc[:, rule_columns].apply(
            pd.to_numeric, errors="raise"
        ).sum(axis=1)
    return joined.drop(columns=rule_columns)


def main() -> None:
    """Build private post-score investigation views through explicit equality joins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-column", default="fused_calibrated")
    parser.add_argument("--as-of-ts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--max-paths", type=int, default=1_000)
    parser.add_argument("--max-branching", type=int, default=5)
    parser.add_argument("--max-high-risk-transactions", type=int, default=100_000)
    args = parser.parse_args()

    joined = _join_private_scores(
        _read_private_parquet(args.transactions),
        _read_private_parquet(args.scores),
        score_column=args.score_column,
    )
    views = build_investigation_views(
        joined,
        as_of_ts=args.as_of_ts,
        score_column=args.score_column,
        window_days=args.window_days,
        risk_threshold=args.risk_threshold,
        top_n=args.top_n,
        max_hops=args.max_hops,
        max_paths=args.max_paths,
        max_branching=args.max_branching,
        max_high_risk_transactions=args.max_high_risk_transactions,
    )
    summary = write_investigation_views(args.output, views)
    manifest = create_run_manifest(
        output_dir=args.output,
        command="aml-build-investigation-views",
        random_seed=0,
        input_paths={"transactions": args.transactions, "scores": args.scores},
        metadata={
            "as_of_ts": views.as_of_ts,
            "window_days": args.window_days,
            "risk_threshold": args.risk_threshold,
            "account_count": summary["account_count"],
            "funds_path_count": summary["funds_path_count"],
            "investigation_case_count": summary["investigation_case_count"],
        },
    )
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "account_count": summary["account_count"],
                "funds_path_count": summary["funds_path_count"],
                "investigation_case_count": summary["investigation_case_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
