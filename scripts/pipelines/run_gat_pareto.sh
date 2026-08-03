#!/usr/bin/env bash
# Six train/validation-only GAT engineering candidates, at most two in parallel.
set -euo pipefail

ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
PY="/data1/yangjuhao/envs/risk/bin/python"
cd "$ROOT"
mkdir -p artifacts/gat_pareto logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1

run_candidate() {
  local name="$1"
  local device="$2"
  local batch="$3"
  local neighbors="$4"
  local window="$5"
  local output="artifacts/gat_pareto/$name"
  local log="logs/gat_pareto_${name}.log"
  if [[ -f "$output/metrics.json" && -f "$output/graphsage.pt" ]]; then
    echo "[$(date -Is)] SKIP complete $name"
    return
  fi
  if [[ -e "$output" ]]; then
    echo "[$(date -Is)] REFUSE partial output $output" >&2
    return 1
  fi
  read -r -a fanout <<< "$neighbors"
  echo "[$(date -Is)] START $name device=$device batch=$batch fanout=$neighbors window=$window"
  "$PY" -u scripts/experiments/run_gat_validation_candidate.py \
    --features artifacts/pit_features \
    --output "$output" \
    --model-config configs/models.gat.yaml \
    --device "$device" \
    --batch-size "$batch" \
    --num-neighbors "${fanout[@]}" \
    --history-window-days "$window" \
    --random-seed 20260722 2>&1 | tee "$log"
  echo "[$(date -Is)] DONE $name"
}

# Each wave changes one factor from the registered baseline: batch=2048,
# fanout=(15,10), history window=30 days. Test data is never read.
run_candidate batch_1024 cuda:3 1024 "15 10" 30 &
pid_a=$!
run_candidate batch_4096 cuda:0 4096 "15 10" 30 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

run_candidate fanout_10_5 cuda:3 2048 "10 5" 30 &
pid_a=$!
run_candidate fanout_25_15 cuda:0 2048 "25 15" 30 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

run_candidate window_14d cuda:3 2048 "15 10" 14 &
pid_a=$!
run_candidate window_60d cuda:0 2048 "15 10" 60 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

echo "[$(date -Is)] ALL GAT PARETO CANDIDATES COMPLETE"
