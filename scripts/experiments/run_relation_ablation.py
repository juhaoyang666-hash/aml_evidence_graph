#!/usr/bin/env python3
"""A3 relation-aware ablation without requiring a full multi-rel RGCN GPU retrain.

1) Slice existing GAT / single-rel RGCN test scores by relation_id
   (cross_border × currency_conversion → {0,1,2,3}).
2) Train a small CPU edge MLP with vs without relation embeddings on downsampled
   PIT edge features (same protocol, honest subsample).
3) Emit docs-ready JSON. Full multi-rel RGCN train (num_relations=4) is:
   python -m aml_evidence_graph.training.run_graphsage \\
     --features artifacts/pit_features --output artifacts/rgcn_rel \\
     --model-config configs/models.rgcn_rel.yaml --device cuda --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.data.splits import TimeSplit
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.graph.snapshots import relation_id_from_frame
from aml_evidence_graph.training.table_baseline import (
    deterministic_negative_downsample,
    load_feature_split,
)

EDGE_FEATURE_CANDIDATES = (
    CANONICAL.amount,
    "amount_log1p",
    "is_cross_border_current_transaction",
    "is_currency_conversion",
    "sender_outgoing_count_1d",
    "sender_outgoing_count_7d",
    "receiver_incoming_count_1d",
    "receiver_incoming_count_7d",
    "relationship_count_7d",
    "relationship_count_30d",
)


def _pick_features(frame: pd.DataFrame) -> list[str]:
    return [c for c in EDGE_FEATURE_CANDIDATES if c in frame.columns]


def _relation_slices(
    scores: pd.DataFrame, features: pd.DataFrame, score_col: str
) -> dict[str, Any]:
    merged = scores.merge(
        features[[CANONICAL.transaction_id, "relation_id"]],
        on=CANONICAL.transaction_id,
        how="inner",
    )
    out: dict[str, Any] = {}
    for rid in sorted(merged["relation_id"].unique()):
        part = merged.loc[merged["relation_id"].eq(rid)]
        y = part[CANONICAL.is_laundering].astype(int)
        if y.nunique() < 2:
            out[str(int(rid))] = {"available": False, "sample_count": int(len(part))}
            continue
        metrics = evaluate_binary_risk_scores(y, part[score_col].clip(0, 1))
        metrics.pop("curves", None)
        out[str(int(rid))] = {"available": True, **metrics}
    return out


class RelationEdgeMLP(nn.Module):
    def __init__(self, n_features: int, n_relations: int, use_relation: bool) -> None:
        super().__init__()
        self.use_relation = use_relation
        self.rel_emb = nn.Embedding(n_relations, 8) if use_relation else None
        in_dim = n_features + (8 if use_relation else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        if self.use_relation and self.rel_emb is not None:
            x = torch.cat([x, self.rel_emb(relation_ids)], dim=-1)
        return self.net(x).squeeze(-1)


def _train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    rel_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    rel_val: np.ndarray,
    *,
    use_relation: bool,
    seed: int,
    epochs: int = 12,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = RelationEdgeMLP(x_train.shape[1], 4, use_relation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - y_train.sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device)
    )
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    rt = torch.tensor(rel_train, dtype=torch.long, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    rv = torch.tensor(rel_val, dtype=torch.long, device=device)

    best_state = None
    best_pr = -1.0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for start in range(0, len(xt), 8192):
            idx = perm[start : start + 8192]
            opt.zero_grad()
            logits = model(xt[idx], rt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(xv, rv)).cpu().numpy()
        pr = float(evaluate_binary_risk_scores(y_val, val_prob)["pr_auc"])
        if pr > best_pr:
            best_pr = pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "val_pr_auc": best_pr}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/relation_ablation"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/pit_features"))
    parser.add_argument(
        "--gat-test",
        type=Path,
        default=Path("artifacts/gat/scores/graphsage_test_scores.parquet"),
    )
    parser.add_argument(
        "--rgcn-test",
        type=Path,
        default=Path("artifacts/rgcn/scores/graphsage_test_scores.parquet"),
    )
    parser.add_argument("--max-train-negatives", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_feature_split(args.features, TimeSplit.TRAIN)
    val = load_feature_split(args.features, TimeSplit.VALIDATION)
    test = load_feature_split(args.features, TimeSplit.TEST)
    for frame in (train, val, test):
        frame["relation_id"] = relation_id_from_frame(frame)

    feature_cols = _pick_features(train)
    train_ds = deterministic_negative_downsample(
        train, maximum_negative_rows=args.max_train_negatives
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_ds.loc[:, feature_cols].to_numpy(dtype=float))
    x_val = scaler.transform(val.loc[:, feature_cols].to_numpy(dtype=float))
    x_test = scaler.transform(test.loc[:, feature_cols].to_numpy(dtype=float))
    y_train = train_ds[CANONICAL.is_laundering].astype(int).to_numpy()
    y_val = val[CANONICAL.is_laundering].astype(int).to_numpy()
    y_test = test[CANONICAL.is_laundering].astype(int).to_numpy()
    r_train = train_ds["relation_id"].to_numpy(dtype=np.int64)
    r_val = val["relation_id"].to_numpy(dtype=np.int64)
    r_test = test["relation_id"].to_numpy(dtype=np.int64)

    results: dict[str, Any] = {
        "protocol": {
            "relation_map": {
                "0": "domestic_same_currency",
                "1": "domestic_conversion",
                "2": "cross_border_same_currency",
                "3": "cross_border_conversion",
            },
            "edge_features": feature_cols,
            "train_rows_after_downsample": int(len(train_ds)),
            "honest_boundary": (
                "CPU relation-embedding MLP + score slices; not a full multi-rel RGCN "
                "neighbor-sampled retrain. Single-rel RGCN (0.903) and GAT (0.948) remain "
                "the neural graph baselines. GPU command for full R=4 RGCN is documented."
            ),
            "full_rgcn_rel_command": (
                "PYTHONPATH=src python -m aml_evidence_graph.training.run_graphsage "
                "--features artifacts/pit_features --output artifacts/rgcn_rel "
                "--model-config configs/models.rgcn_rel.yaml --device cuda --max-gpus 4 --overwrite"
            ),
        },
        "relation_counts_test": {
            str(int(k)): int(v) for k, v in test["relation_id"].value_counts().sort_index().items()
        },
    }

    gat = pd.read_parquet(args.gat_test)
    rgcn = pd.read_parquet(args.rgcn_test)
    results["gat_relation_slices"] = _relation_slices(gat, test, "graphsage")
    results["rgcn_relation_slices"] = _relation_slices(rgcn, test, "graphsage")

    for name, use_rel in (("mlp_no_relation", False), ("mlp_with_relation", True)):
        packed = _train_mlp(
            x_train, y_train, r_train, x_val, y_val, r_val, use_relation=use_rel, seed=args.seed
        )
        model: RelationEdgeMLP = packed["model"]
        with torch.no_grad():
            test_prob = torch.sigmoid(
                model(
                    torch.tensor(x_test, dtype=torch.float32),
                    torch.tensor(r_test, dtype=torch.long),
                )
            ).numpy()
        metrics = evaluate_binary_risk_scores(y_test, test_prob)
        metrics.pop("curves", None)
        results[name] = {
            "validation_pr_auc": packed["val_pr_auc"],
            "test_metrics": metrics,
        }

    results["verdict"] = {
        "mlp_with_relation_pr_auc": results["mlp_with_relation"]["test_metrics"]["pr_auc"],
        "mlp_no_relation_pr_auc": results["mlp_no_relation"]["test_metrics"]["pr_auc"],
        "relation_embedding_helps": bool(
            results["mlp_with_relation"]["test_metrics"]["pr_auc"]
            > results["mlp_no_relation"]["test_metrics"]["pr_auc"] + 1e-4
        ),
        "reference_rgcn_single_rel": 0.9031,
        "reference_gat": 0.9483,
    }
    (args.output_dir / "relation_ablation_summary.json").write_text(
        json.dumps(results, indent=2, default=float) + "\n"
    )
    print(json.dumps(results["verdict"] | {"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
