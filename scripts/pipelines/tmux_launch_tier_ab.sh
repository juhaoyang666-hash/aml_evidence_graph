#!/usr/bin/env bash
# Launch remaining Tier A3 / B experiments in detached tmux sessions.
set -euo pipefail
ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
PY="/data1/yangjuhao/envs/risk/bin/python"
export PATH="/data1/yangjuhao/envs/risk/bin:$PATH"
export PYTHONPATH=src
cd "$ROOT"
mkdir -p logs artifacts

launch() {
  local name="$1"
  shift
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "session $name already exists; skip"
    return
  fi
  tmux new-session -d -s "$name" "cd '$ROOT'; export PYTHONPATH=src PATH='/data1/yangjuhao/envs/risk/bin:'\"\$PATH\"; $* ; echo DONE_\$? | tee -a logs/${name}.log; exec bash"
  echo "started $name"
}

# A3 relation ablation (CPU) — hours for full PIT load + MLP
launch aml_a3_rel "$PY scripts/experiments/run_relation_ablation.py 2>&1 | tee logs/relation_ablation.log"

# B2 GAT distill CatBoost — hours
launch aml_b2_distill "$PY scripts/experiments/run_gat_distill_catboost.py 2>&1 | tee logs/gat_distill.log"

# B1 sequence GRU — hours (loads full prepared splits)
launch aml_b1_seq "$PY scripts/experiments/run_sequence_baseline.py 2>&1 | tee logs/sequence_baseline.log"

# B4 batch replay — minutes–1h
launch aml_b4_batch "$PY scripts/data/run_batch_feature_replay.py 2>&1 | tee logs/batch_feature_replay.log"

# B5 golden template (no LLM) — minutes
launch aml_b5_golden "$PY -m aml_evidence_graph.investigation.golden --cases golden/cases_v1.json --typologies knowledge/typologies --output artifacts/golden_summary.json 2>&1 | tee logs/golden_expanded.log"

# A3 full multi-rel RGCN when CUDA appears (will fail fast if no GPU)
launch aml_a3_rgcn_rel "$PY -m aml_evidence_graph.training.run_graphsage --features artifacts/pit_features --output artifacts/rgcn_rel --model-config configs/models.rgcn_rel.yaml --device cuda --max-gpus 4 --overwrite 2>&1 | tee logs/rgcn_rel.log || echo 'RGCN_REL_SKIPPED_NO_GPU'"

tmux ls
