#!/usr/bin/env python3
"""Unsupervised anomaly baselines on PIT numeric features (train-fit, test-score).

Fits Isolation Forest (and optional shallow Autoencoder) using train-period rows
only — default: negatives-only (semi-supervised one-class) — then reports test
PR-AUC / budget points beside supervised CatBoost/GAT without replacing them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores

META_COLUMNS = {
    CANONICAL.transaction_id,
    CANONICAL.event_ts,
    CANONICAL.sender_account_id,
    CANONICAL.receiver_account_id,
    CANONICAL.payment_currency,
    CANONICAL.received_currency,
    CANONICAL.sender_location,
    CANONICAL.receiver_location,
    CANONICAL.payment_type,
    CANONICAL.is_laundering,
    CANONICAL.laundering_type,
    "source_row_number",
    "rule_hits",
}


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = []
    for column in frame.columns:
        if column in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            cols.append(column)
    if not cols:
        raise ValueError("No numeric PIT feature columns found.")
    return cols


def _iter_split_partitions(root: Path, split: str) -> list[Path]:
    return sorted(root.glob(f"event_date=*/split={split}/*.parquet"))


def _load_sample(
    root: Path,
    *,
    split: str,
    feature_columns: list[str] | None,
    max_rows: int,
    seed: int,
    negatives_only: bool,
) -> tuple[pd.DataFrame, list[str]]:
    paths = _iter_split_partitions(root, split)
    if not paths:
        raise FileNotFoundError(f"No partitions for split={split} under {root}")
    rng = np.random.default_rng(seed)
    # Reservoir-style approximate sample across partitions.
    collected: list[pd.DataFrame] = []
    seen = 0
    columns = None
    for path in paths:
        frame = pd.read_parquet(path)
        if columns is None:
            columns = feature_columns or _feature_columns(frame)
        if negatives_only:
            frame = frame.loc[frame[CANONICAL.is_laundering].astype(int).eq(0)]
        if frame.empty:
            continue
        keep = [CANONICAL.transaction_id, CANONICAL.is_laundering, *columns]
        frame = frame.loc[:, keep]
        remaining = max_rows - seen
        if remaining <= 0:
            break
        if len(frame) > remaining:
            idx = rng.choice(len(frame), size=remaining, replace=False)
            frame = frame.iloc[idx]
        collected.append(frame)
        seen += len(frame)
        if seen >= max_rows:
            break
    assert columns is not None
    out = pd.concat(collected, ignore_index=True)
    return out, columns


def _scores_from_decision(decision: np.ndarray) -> np.ndarray:
    """Map IsolationForest decision_function (higher=more normal) to [0,1] anomaly score."""
    # decision_function: higher = more normal. Invert and min-max to [0,1].
    anomaly = -decision
    lo, hi = float(anomaly.min()), float(anomaly.max())
    if hi <= lo:
        return np.zeros_like(anomaly, dtype=float)
    return (anomaly - lo) / (hi - lo)


def _fit_autoencoder(
    x_train: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[Any, Any]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = x_train.shape[1]
    hidden = max(8, n_features // 4)
    model = nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, max(4, hidden // 2)),
        nn.ReLU(),
        nn.Linear(max(4, hidden // 2), hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_features),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(tensor), device=device)
        for start in range(0, len(tensor), batch_size):
            batch = tensor[perm[start : start + batch_size]]
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
    model.eval()
    return model, device


def _autoencoder_scores(model: Any, device: Any, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device=device)
        recon = model(tensor)
        err = ((recon - tensor) ** 2).mean(dim=1).detach().cpu().numpy()
    lo, hi = float(err.min()), float(err.max())
    if hi <= lo:
        return np.zeros_like(err, dtype=float)
    return (err - lo) / (hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/unsupervised_baseline"))
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/pit_features"))
    parser.add_argument("--train-max-rows", type=int, default=400_000)
    parser.add_argument("--test-max-rows", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--autoencoder", action="store_true", default=True)
    parser.add_argument("--no-autoencoder", action="store_false", dest="autoencoder")
    parser.add_argument("--ae-epochs", type=int, default=8)
    parser.add_argument("--ae-batch-size", type=int, default=4096)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_frame, feature_columns = _load_sample(
        args.feature_root,
        split="train",
        feature_columns=None,
        max_rows=args.train_max_rows,
        seed=args.seed,
        negatives_only=True,
    )
    test_frame, _ = _load_sample(
        args.feature_root,
        split="test",
        feature_columns=feature_columns,
        max_rows=args.test_max_rows,
        seed=args.seed + 1,
        negatives_only=False,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(
        train_frame.loc[:, feature_columns].to_numpy(dtype=float)
    )
    x_test = scaler.transform(test_frame.loc[:, feature_columns].to_numpy(dtype=float))
    y_test = test_frame[CANONICAL.is_laundering].astype(int).to_numpy()

    forest = IsolationForest(
        n_estimators=args.n_estimators,
        contamination="auto",
        random_state=args.seed,
        n_jobs=-1,
    )
    forest.fit(x_train)
    if_scores = _scores_from_decision(forest.decision_function(x_test))
    if_metrics = evaluate_binary_risk_scores(y_test, if_scores)
    if_metrics.pop("curves", None)

    results: dict[str, Any] = {
        "protocol": {
            "feature_root": str(args.feature_root),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "feature_count": len(feature_columns),
            "train_negatives_only": True,
            "sampling": "partition-sequential cap (not full scan)",
            "honest_boundary": (
                "Sampled PIT partitions; unsupervised scores are a parallel baseline, "
                "not a replacement for CatBoost/GAT. Synthetic SAML-D; not production."
            ),
        },
        "isolation_forest": {
            "n_estimators": args.n_estimators,
            "metrics": if_metrics,
        },
    }

    score_frame = pd.DataFrame(
        {
            CANONICAL.transaction_id: test_frame[CANONICAL.transaction_id],
            CANONICAL.is_laundering: y_test,
            "isolation_forest": if_scores,
        }
    )

    if args.autoencoder:
        model, device = _fit_autoencoder(
            x_train,
            epochs=args.ae_epochs,
            batch_size=args.ae_batch_size,
            seed=args.seed,
        )
        ae_scores = _autoencoder_scores(model, device, x_test)
        ae_metrics = evaluate_binary_risk_scores(y_test, ae_scores)
        ae_metrics.pop("curves", None)
        results["autoencoder"] = {
            "epochs": args.ae_epochs,
            "batch_size": args.ae_batch_size,
            "device": str(device),
            "metrics": ae_metrics,
        }
        score_frame["autoencoder"] = ae_scores

    score_frame.to_parquet(args.output_dir / "unsupervised_test_scores.parquet", index=False)
    (args.output_dir / "unsupervised_summary.json").write_text(
        json.dumps(results, indent=2, default=float) + "\n"
    )
    (args.output_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "isolation_forest_pr_auc": if_metrics["pr_auc"],
                "autoencoder_pr_auc": (
                    results.get("autoencoder", {}).get("metrics", {}).get("pr_auc")
                ),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
