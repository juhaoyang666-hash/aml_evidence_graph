#!/usr/bin/env python3
"""B1: account outgoing-sequence GRU baseline (edge-level laundering prediction).

For each transaction, take the sender's last K prior outgoing events (strictly
before t) from prepared_transactions, encode [log1p(amount), Δt_hours,
payment_type_id, cross_border, currency_mismatch], run a GRU, and score the
current edge. Train negatives are hash-downsampled. Parallel to CatBoost/GAT —
does not replace them.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch import nn

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.data.splits import TimeSplit
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores


def _load_prepared(root: Path, split: TimeSplit) -> pd.DataFrame:
    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    frame = dataset.to_table(filter=ds.field("split") == split.value).to_pandas()
    if frame.empty:
        raise ValueError(f"No prepared rows for split={split.value}")
    frame[CANONICAL.event_ts] = pd.to_datetime(frame[CANONICAL.event_ts], utc=True)
    return frame.sort_values(
        [CANONICAL.event_ts, CANONICAL.source_row_number], kind="stable"
    ).reset_index(drop=True)


def _fit_payment_vocab(train: pd.DataFrame) -> dict[str, int]:
    values = sorted(train[CANONICAL.payment_type].astype(str).fillna("__NA__").unique())
    return {v: i + 1 for i, v in enumerate(values)}  # 0 = unknown


def _encode_event(
    amount: float,
    delta_hours: float,
    payment_id: int,
    cross_border: float,
    conversion: float,
) -> list[float]:
    return [
        float(np.log1p(max(amount, 0.0))),
        float(np.log1p(max(delta_hours, 0.0))),
        float(payment_id),
        float(cross_border),
        float(conversion),
    ]


def build_sequence_tensors(
    frame: pd.DataFrame,
    *,
    payment_vocab: dict[str, int],
    seq_len: int,
    max_negatives: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = frame[CANONICAL.is_laundering].astype(int)
    if max_negatives is not None:
        pos = frame.loc[labels.eq(1)]
        neg = frame.loc[labels.eq(0)].copy()
        if len(neg) > max_negatives:
            keys = (neg[CANONICAL.source_row_number].astype(np.int64) * 2654435761) % (10**9 + 7)
            neg = neg.assign(_h=keys).nsmallest(max_negatives, "_h").drop(columns="_h")
        frame = pd.concat([pos, neg], ignore_index=True).sort_values(
            [CANONICAL.event_ts, CANONICAL.source_row_number], kind="stable"
        )

    hist_feats: dict[str, deque] = defaultdict(lambda: deque(maxlen=seq_len))
    hist_ts: dict[str, deque] = defaultdict(lambda: deque(maxlen=seq_len))
    xs: list[np.ndarray] = []
    ys: list[int] = []
    lengths: list[int] = []
    feat_dim = 5

    for row in frame.itertuples(index=False):
        sender = str(getattr(row, CANONICAL.sender_account_id))
        ts = pd.Timestamp(getattr(row, CANONICAL.event_ts))
        amount = float(getattr(row, CANONICAL.amount))
        payment = str(getattr(row, CANONICAL.payment_type) or "__NA__")
        pay_id = payment_vocab.get(payment, 0)
        cross = float(
            str(getattr(row, CANONICAL.sender_location))
            != str(getattr(row, CANONICAL.receiver_location))
        )
        conversion = float(
            str(getattr(row, CANONICAL.payment_currency))
            != str(getattr(row, CANONICAL.received_currency))
        )
        seq = np.zeros((seq_len, feat_dim), dtype=np.float32)
        feats = list(hist_feats[sender])
        length = len(feats)
        if length:
            seq[-length:] = np.asarray(feats, dtype=np.float32)
        xs.append(seq)
        ys.append(int(getattr(row, CANONICAL.is_laundering)))
        lengths.append(length)

        prev_times = list(hist_ts[sender])
        delta_hours = 0.0
        if prev_times:
            delta_hours = max((ts - prev_times[-1]).total_seconds() / 3600.0, 0.0)
        encoded = _encode_event(amount, delta_hours, pay_id, cross, conversion)
        hist_feats[sender].append(encoded)
        hist_ts[sender].append(ts)

    return np.stack(xs), np.asarray(ys, dtype=np.int64), np.asarray(lengths, dtype=np.int64)


class SeqGRU(nn.Module):
    def __init__(self, input_dim: int = 5, hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: [B, K, F]; use last hidden after packing
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.head(h[-1]).squeeze(-1)


def _train_eval(
    x_tr, y_tr, len_tr, x_va, y_va, len_va, x_te, y_te, len_te, *, seed: int, epochs: int
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = SeqGRU().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos = max(int(y_tr.sum()), 1)
    neg = max(int(len(y_tr) - y_tr.sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))

    def tensors(x, y, leng):
        return (
            torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device),
            torch.tensor(np.maximum(leng, 1), dtype=torch.long, device=device),
        )

    xt, yt, lt = tensors(x_tr, y_tr, len_tr)
    xv, yv, lv = tensors(x_va, y_va, len_va)
    xe, ye, le = tensors(x_te, y_te, len_te)

    best_state, best_pr = None, -1.0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for start in range(0, len(xt), 1024):
            idx = perm[start : start + 1024]
            # zero-length sequences: still valid via clamp
            opt.zero_grad()
            logits = model(xt[idx], lt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(xv, lv)).cpu().numpy()
        pr = float(evaluate_binary_risk_scores(y_va, vp)["pr_auc"])
        if pr > best_pr:
            best_pr = pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(xe, le)).cpu().numpy()
    metrics = evaluate_binary_risk_scores(y_te, tp)
    metrics.pop("curves", None)
    return {"validation_pr_auc": best_pr, "test_metrics": metrics, "scores": tp}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/prepared_transactions"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sequence_baseline"))
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-train-negatives", type=int, default=200_000)
    parser.add_argument("--max-val-rows", type=int, default=200_000)
    parser.add_argument("--max-test-rows", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = _load_prepared(args.prepared, TimeSplit.TRAIN)
    val = _load_prepared(args.prepared, TimeSplit.VALIDATION)
    test = _load_prepared(args.prepared, TimeSplit.TEST)
    vocab = _fit_payment_vocab(train)

    # For val/test downsampling keep all positives + hash negatives for speed
    def prep(frame, max_neg):
        return build_sequence_tensors(
            frame,
            payment_vocab=vocab,
            seq_len=args.seq_len,
            max_negatives=max_neg,
            seed=args.seed,
        )

    x_tr, y_tr, l_tr = prep(train, args.max_train_negatives)
    x_va, y_va, l_va = prep(val, args.max_val_rows)
    x_te, y_te, l_te = prep(test, args.max_test_rows)

    result = _train_eval(
        x_tr, y_tr, l_tr, x_va, y_va, l_va, x_te, y_te, l_te, seed=args.seed, epochs=args.epochs
    )
    summary = {
        "protocol": {
            "seq_len": args.seq_len,
            "train_rows": int(len(y_tr)),
            "val_rows": int(len(y_va)),
            "test_rows": int(len(y_te)),
            "test_positive_rate": float(y_te.mean()),
            "honest_boundary": (
                "Sender outgoing history GRU on downsampled prepared txs; CPU; "
                "not full-universe sequence model. Parallel to CatBoost/GAT."
            ),
        },
        "validation_pr_auc": result["validation_pr_auc"],
        "test_metrics": result["test_metrics"],
        "reference_catboost": 0.8092,
        "reference_gat": 0.9483,
    }
    pd.DataFrame(
        {
            "is_laundering": y_te,
            "sequence_gru": result["scores"],
        }
    ).to_parquet(args.output_dir / "sequence_test_scores_sample.parquet", index=False)
    (args.output_dir / "sequence_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    print(
        json.dumps(
            {
                "test_pr_auc": summary["test_metrics"]["pr_auc"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
