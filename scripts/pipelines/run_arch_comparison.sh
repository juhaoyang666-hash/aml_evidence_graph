#!/usr/bin/env bash
# Sequential full-protocol GAT / RGCN / PNA training (same PIT features + models.yaml hyperparams).
# Does not overwrite artifacts/graphsage.
# Skips any arch whose artifacts/<arch>/metrics.json already exists unless FORCE_RETRAIN=1.
set -euo pipefail
ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
cd "$ROOT"
mkdir -p logs
export PATH="/data1/yangjuhao/envs/risk/bin:$HOME/.local/bin:/usr/bin:$PATH"
export PY=/data1/yangjuhao/envs/risk/bin/python
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-aml
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAX_GPUS="${MAX_GPUS:-4}"

log() { echo "[$(date -Is)] $*"; }

summarize_metrics() {
  local arch="$1"
  local out="$2"
  # Heredoc avoids nested quote breakage in inline python -c strings.
  "$PY" - "$arch" "$out" <<'PY'
import json
import sys

arch, out = sys.argv[1], sys.argv[2]
m = json.load(open(f"{out}/metrics.json"))
tm = m["test_metrics"]
b = tm["alert_budgets"].get("0.1000%", {})
rt = m.get("runtime", {})
print(
    arch,
    "pr_auc",
    tm["pr_auc"],
    "p@0.1%",
    b.get("precision_at_k"),
    "r@0.1%",
    b.get("recall_at_k"),
    "test_rps",
    rt.get("test_rows_per_second"),
    "wall_s",
    rt.get("wall_time_seconds"),
)
PY
}

log "env check"
$PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count(), "CUDA_VISIBLE_DEVICES", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))'

run_arch() {
  local arch="$1"
  local config="$2"
  local out="artifacts/${arch}"
  if [[ -f "${out}/metrics.json" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
    log "SKIP ${arch} (metrics.json exists; set FORCE_RETRAIN=1 to overwrite)"
    summarize_metrics "$arch" "$out" || true
    return 0
  fi
  log "START ${arch} -> ${out}"
  # Line-buffered tee so logs are inspectable while detached from tmux.
  stdbuf -oL -eL "$PY" -u -m aml_evidence_graph.training.run_graphsage \
    --features artifacts/pit_features \
    --output "${out}" \
    --model-config "${config}" \
    --device cuda \
    --max-gpus "$MAX_GPUS" \
    --overwrite \
    2>&1 | stdbuf -oL -eL tee "logs/${arch}_full.log"
  log "OK ${arch}"
  summarize_metrics "$arch" "$out"
}

run_arch gat configs/models.gat.yaml
run_arch rgcn configs/models.rgcn.yaml
run_arch pna configs/models.pna.yaml

log "ARCH COMPARISON COMPLETE"
date -Is > logs/arch_comparison_done.flag
