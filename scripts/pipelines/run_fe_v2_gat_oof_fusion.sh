#!/usr/bin/env bash
# fe_v2 main line: GAT train -> table OOF -> GAT OOF -> catboost+GAT fusion -> test eval.
# Writes only to *_fe_v2 sidecar dirs; does NOT overwrite historical
# artifacts/{pit_features,gat,table_oof,graph_oof_gat,fusion_cb_gat,*}.
set -euo pipefail

ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
cd "$ROOT"
mkdir -p logs artifacts/logs

export PATH="/data1/yangjuhao/envs/risk/bin:$HOME/.local/bin:/usr/bin:$PATH"
PY=/data1/yangjuhao/envs/risk/bin/python
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-aml
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAX_GPUS="${MAX_GPUS:-4}"

FEATURES="artifacts/pit_features_fe_v2"
MODEL_CFG="configs/models.gat.yaml"
TABLE_CFG="configs/models.yaml"
GAT_OUT="artifacts/gat_fe_v2"
TABLE_OOF_OUT="artifacts/table_oof_fe_v2"
GRAPH_OOF_OUT="artifacts/graph_oof_fe_v2"
FUSION_OUT="artifacts/fusion_fe_v2"
FUSION_TEST_OUT="artifacts/fusion_test_fe_v2"
TABLE_BASE="artifacts/table_baseline_fe_v2"
STATUS="artifacts/logs/fe_v2_gat_oof_fusion_status.json"
CHAIN_LOG="logs/fe_v2_gat_oof_fusion.log"

log() { echo "[$(date -Is)] $*" | tee -a "$CHAIN_LOG"; }

write_status() {
  local step="$1"
  local state="$2"
  local detail="${3:-}"
  local started="${4:-}"
  local finished="${5:-}"
  "$PY" - "$STATUS" "$step" "$state" "$detail" "$started" "$finished" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

path, step, state, detail, started, finished = sys.argv[1:7]
p = Path(path)
prev = {}
if p.is_file():
    try:
        prev = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        prev = {}
steps = dict(prev.get("steps") or {})
entry = dict(steps.get(step) or {})
entry["state"] = state
if detail:
    entry["detail"] = detail
if started:
    entry["started_at"] = started
if finished:
    entry["finished_at"] = finished
if started and finished:
    try:
        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(finished)
        entry["duration_seconds"] = (t1 - t0).total_seconds()
    except Exception:
        pass
now = datetime.now(timezone.utc).isoformat(timespec="seconds")
entry["updated_at"] = now
steps[step] = entry
payload = {
    "pipeline": "fe_v2_gat_oof_fusion",
    "features": "artifacts/pit_features_fe_v2",
    "model_config_gat": "configs/models.gat.yaml",
    "outputs": {
        "gat": "artifacts/gat_fe_v2",
        "table_oof": "artifacts/table_oof_fe_v2",
        "graph_oof": "artifacts/graph_oof_fe_v2",
        "fusion": "artifacts/fusion_fe_v2",
        "fusion_test": "artifacts/fusion_test_fe_v2",
    },
    "current_step": step,
    "current_state": state,
    "updated_at": now,
    "steps": steps,
}
if state == "failed":
    payload["failed_step"] = step
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

run_step() {
  local name="$1"
  shift
  local start
  start="$(date -Is)"
  write_status "$name" "running" "$*" "$start" ""
  log "START ${name}"
  if "$@"; then
    local end
    end="$(date -Is)"
    write_status "$name" "ok" "" "$start" "$end"
    log "OK ${name}"
  else
    local code=$?
    local end
    end="$(date -Is)"
    write_status "$name" "failed" "exit=${code}" "$start" "$end"
    log "FAIL ${name} exit=${code}"
    return "$code"
  fi
}

log "===== fe_v2 GAT → OOF → fusion START ====="
log "env check"
$PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count(), "CUDA_VISIBLE_DEVICES", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))' | tee -a "$CHAIN_LOG"

if [[ ! -f "${FEATURES}/_run_manifest.json" ]]; then
  log "ERROR: missing ${FEATURES}/_run_manifest.json"
  exit 1
fi
if [[ ! -f "${TABLE_BASE}/metrics.json" ]]; then
  log "ERROR: missing ${TABLE_BASE}/metrics.json (need fe_v2 CatBoost scores for fusion)"
  exit 1
fi
if [[ ! -f "${TABLE_BASE}/scores/table_validation_scores.parquet" || ! -f "${TABLE_BASE}/scores/table_test_scores.parquet" ]]; then
  log "ERROR: missing table_baseline_fe_v2 score parquets"
  exit 1
fi

# 1) GAT train
if [[ -f "${GAT_OUT}/metrics.json" && -f "${GAT_OUT}/scores/graphsage_test_scores.parquet" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
  log "SKIP gat_fe_v2 (metrics present; FORCE_RETRAIN=1 to overwrite)"
  write_status "gat" "skipped" "metrics present" "" ""
else
  run_step gat stdbuf -oL -eL "$PY" -u -m aml_evidence_graph.training.run_graphsage \
    --features "$FEATURES" \
    --output "$GAT_OUT" \
    --model-config "$MODEL_CFG" \
    --device cuda \
    --max-gpus "$MAX_GPUS" \
    --overwrite
fi

# 2) table OOF (fe_v2 features; do not reuse v1 table_oof)
if [[ -f "${TABLE_OOF_OUT}/table_oof_scores.parquet" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
  log "SKIP table_oof_fe_v2"
  write_status "table_oof" "skipped" "parquet present" "" ""
else
  run_step table_oof stdbuf -oL -eL "$PY" -u -m aml_evidence_graph.training.oof \
    --features "$FEATURES" \
    --output "$TABLE_OOF_OUT" \
    --model table \
    --model-config "$TABLE_CFG" \
    --splits 3 \
    --minimum-training-months 2 \
    --overwrite
fi

# 3) GAT OOF
if [[ -f "${GRAPH_OOF_OUT}/graphsage_oof_scores.parquet" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
  log "SKIP graph_oof_fe_v2"
  write_status "graph_oof" "skipped" "parquet present" "" ""
else
  run_step graph_oof stdbuf -oL -eL "$PY" -u -m aml_evidence_graph.training.oof \
    --features "$FEATURES" \
    --output "$GRAPH_OOF_OUT" \
    --model graphsage \
    --model-config "$MODEL_CFG" \
    --splits 3 \
    --minimum-training-months 2 \
    --device cuda \
    --max-gpus "$MAX_GPUS" \
    --overwrite
fi

# 4) fusion train (catboost + GAT; component column still named graphsage)
if [[ -f "${FUSION_OUT}/run_manifest.json" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
  log "SKIP fusion_fe_v2"
  write_status "fusion" "skipped" "manifest present" "" ""
else
  run_step fusion stdbuf -oL -eL "$PY" -u -m aml_evidence_graph.training.fusion \
    --oof "${TABLE_OOF_OUT}/table_oof_scores.parquet" \
    --oof "${GRAPH_OOF_OUT}/graphsage_oof_scores.parquet" \
    --validation "${TABLE_BASE}/scores/table_validation_scores.parquet" \
    --validation "${GAT_OUT}/scores/graphsage_validation_scores.parquet" \
    --components catboost,graphsage \
    --output "$FUSION_OUT" \
    --model-config "$TABLE_CFG" \
    --overwrite
fi

# 5) frozen test eval
if [[ -f "${FUSION_TEST_OUT}/metrics.json" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
  log "SKIP fusion_test_fe_v2"
  write_status "fusion_test" "skipped" "metrics present" "" ""
else
  run_step fusion_test stdbuf -oL -eL "$PY" -u -m aml_evidence_graph.training.evaluate_fusion \
    --fusion-dir "$FUSION_OUT" \
    --test "${TABLE_BASE}/scores/table_test_scores.parquet" \
    --test "${GAT_OUT}/scores/graphsage_test_scores.parquet" \
    --features "$FEATURES" \
    --output "$FUSION_TEST_OUT" \
    --overwrite
fi

# 6) summary comparison
log "SUMMARY"
"$PY" - <<'PY' | tee -a "$CHAIN_LOG"
import json
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

table = load("artifacts/table_baseline_fe_v2/metrics.json")
cb = table["test_metrics"]["catboost"]
gat = load("artifacts/gat_fe_v2/metrics.json")
fus = load("artifacts/fusion_test_fe_v2/metrics.json")
gtm = gat["test_metrics"]
ftm = fus["test_metrics"]

main_gat = 0.9483
main_fus = 0.9175
rows = [
    ("CatBoost(fe_v2)", cb["pr_auc"], cb.get("roc_auc"), None),
    ("GAT(fe_v2)", gtm["pr_auc"], gtm.get("roc_auc"), gtm["pr_auc"] - main_gat),
    ("fusion(fe_v2)", ftm["pr_auc"], ftm.get("roc_auc"), ftm["pr_auc"] - main_fus),
]
print("metric_table:")
for name, pr, roc, delta in rows:
    d = f" delta={delta:+.4f}" if delta is not None else ""
    print(f"  {name}: PR-AUC={pr:.6f} ROC-AUC={roc:.6f}{d}")
print(f"vs v1 historical GAT 0.9483: GAT_fe_v2 Δ={gtm['pr_auc']-main_gat:+.4f}")
print(f"vs v1 historical fusion 0.9175: fusion_fe_v2 Δ={ftm['pr_auc']-main_fus:+.4f}")
print(f"fusion_exceeds_gat_alone: {ftm['pr_auc'] > gtm['pr_auc']}")
print(f"best_system_is_fusion: {ftm['pr_auc'] >= max(cb['pr_auc'], gtm['pr_auc'])}")
print(f"gat_run_id={gat.get('run_id')} fusion_test_run_id={fus.get('run_id')} table_run_id={table.get('run_id')}")
print(f"edge_feature_count={len(gat.get('edge_feature_columns', []))}")
PY

write_status "pipeline" "ok" "complete" "" "$(date -Is)"
log "===== fe_v2 GAT → OOF → fusion COMPLETE ====="
date -Is > logs/fe_v2_gat_oof_fusion_done.flag
