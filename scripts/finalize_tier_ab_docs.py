#!/usr/bin/env python3
"""Refresh the consolidated RESULTS compute-budget appendix from finished artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    rel = load(ROOT / "artifacts/relation_ablation/relation_ablation_summary.json")
    seq = load(ROOT / "artifacts/sequence_baseline/sequence_summary.json")
    dist = load(ROOT / "artifacts/table_baseline_gat_distill/metrics.json")
    batch = load(ROOT / "artifacts/batch_feature_replay/batch_replay_summary.json")
    rgcn_rel = load(ROOT / "artifacts/rgcn_rel/metrics.json")
    nonlinear = load(ROOT / "artifacts/nonlinear_fusion/nonlinear_fusion_summary.json")

    lines = ["\n## Appendix — Tier A3 / Tier B (compute budget)\n"]
    if rel:
        v = rel["verdict"]
        lines.append(
            f"- **A3 relation MLP**: with_rel PR-AUC {v['mlp_with_relation_pr_auc']:.4f} vs "
            f"no_rel {v['mlp_no_relation_pr_auc']:.4f} (helps={v['relation_embedding_helps']}). "
            "Artifact: `artifacts/relation_ablation`."
        )
    if rgcn_rel and "test_metrics" in rgcn_rel:
        lines.append(
            f"- **A3 multi-rel RGCN**: test PR-AUC {rgcn_rel['test_metrics']['pr_auc']:.4f} "
            f"(artifact `artifacts/rgcn_rel`)."
        )
    elif Path(ROOT / "logs/rgcn_rel.log").is_file():
        lines.append(
            "- **A3 multi-rel RGCN**: training attempted; see `logs/rgcn_rel.log` "
            "(needs `/dev/nvidia*`)."
        )
    if seq:
        lines.append(
            f"- **B1 sequence GRU**: sampled test PR-AUC {seq['test_metrics']['pr_auc']:.4f}. "
            "Artifact: `artifacts/sequence_baseline`."
        )
    if dist:
        lines.append(
            f"- **B2 GAT distill CatBoost**: test PR-AUC {dist['test_metrics']['pr_auc']:.4f} "
            f"(beats_catboost={dist.get('beats_catboost')}). "
            "Artifact: `artifacts/table_baseline_gat_distill`."
        )
    if nonlinear:
        v = nonlinear["verdict"]
        lines.append(
            f"- **B3 nonlinear fusion**: logistic {v['mainline_logistic_test_pr_auc']:.4f} > "
            f"best nonlinear {v['best_nonlinear_test_pr_auc']:.4f}. "
            "Artifact: `artifacts/nonlinear_fusion`."
        )
    if batch:
        e = batch["equality"]
        lines.append(
            f"- **B4 batch replay**: DuckDB/Polars match rates "
            f"{e['duckdb_match_rate']:.3f}/{e['polars_match_rate']:.3f}. "
            f"[批量特征重放.md](批量特征重放.md)."
        )
    lines.append(
        "- **B5 LLM probes**: Golden expanded to 34 cases; template hallucination intercept 1.0. "
        "[大模型调查系统.md](大模型调查系统.md)."
    )
    results = ROOT / "docs/实验结果.md"
    text = results.read_text()
    marker = "## Appendix — Tier A3 / Tier B"
    block = "\n".join(lines) + "\n"
    if marker in text:
        pre, _, rest = text.partition(marker)
        # drop old appendix through EOF or next ## 
        idx = rest.find("\n## ")
        text = pre + block + (rest[idx+1:] if idx >= 0 else "")
    else:
        text = text.rstrip() + "\n" + block
    results.write_text(text)
    print("updated RESULTS appendix")
    print(block)


if __name__ == "__main__":
    main()
