#!/usr/bin/env bash
# Four validation-only leave-one-family-out runs, at most two in parallel.
set -euo pipefail

ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
PY="/data1/yangjuhao/envs/risk/bin/python"
cd "$ROOT"
mkdir -p artifacts/gat_feature_ablation logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1

run_ablation() {
  local family="$1"
  local device="$2"
  local output="artifacts/gat_feature_ablation/without_${family}"
  local log="logs/gat_feature_ablation_without_${family}.log"
  if [[ -f "$output/metrics.json" && -f "$output/graphsage.pt" ]]; then
    echo "[$(date -Is)] SKIP complete without_${family}"
    return
  fi
  if [[ -e "$output" ]]; then
    echo "[$(date -Is)] REFUSE partial output $output" >&2
    return 1
  fi
  echo "[$(date -Is)] START without_${family} device=${device}"
  "$PY" -u scripts/run_gat_validation_candidate.py \
    --features artifacts/pit_features \
    --output "$output" \
    --model-config configs/models.gat.yaml \
    --device "$device" \
    --batch-size 2048 \
    --num-neighbors 15 10 \
    --history-window-days 60 \
    --random-seed 20260722 \
    --exclude-feature-family "$family" 2>&1 | tee "$log"
  echo "[$(date -Is)] DONE without_${family}"
}

run_ablation amount cuda:3 &
pid_a=$!
run_ablation temporal_behavior cuda:0 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

run_ablation relationship cuda:3 &
pid_a=$!
run_ablation node_stats cuda:0 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

echo "[$(date -Is)] ALL GAT FEATURE ABLATIONS COMPLETE"
