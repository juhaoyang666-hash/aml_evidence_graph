#!/usr/bin/env bash
# Re-run table_baseline against the rule-backfilled PIT dataset so that
# alert_reduction_vs_rules is populated. Writes to a separate output directory so
# the existing artifacts/table_baseline stays intact until the new run is verified.
set -euo pipefail

ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
cd "$ROOT"
export PATH="/data1/yangjuhao/envs/risk/bin:$HOME/.local/bin:/usr/bin:$PATH"
export PY="/data1/yangjuhao/envs/risk/bin/python"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-aml}"
# table_baseline is CPU-bound; leave headroom for the concurrent OOF job.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"

OUT="${OUT:-artifacts/table_baseline_rules}"
LOG="logs/table_baseline_rules.log"
mkdir -p logs artifacts

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "precheck: rule columns present in PIT dataset"
"$PY" - <<'PY' | tee -a "$LOG"
import pyarrow.dataset as ds
d = ds.dataset("artifacts/pit_features", format="parquet", partitioning="hive")
rule_columns = [n for n in d.schema.names if n.startswith("rule_") and n.endswith("_hit")]
if not rule_columns:
    raise SystemExit("No rule_*_hit columns; run scripts/data/backfill_rule_hits.py first.")
print("rule_columns:", rule_columns)
PY

log "START table_baseline -> ${OUT}"
set +e
stdbuf -oL -eL "$PY" -m aml_evidence_graph.training.table_baseline \
  --features artifacts/pit_features \
  --output "$OUT" \
  --model-config configs/models.yaml \
  --overwrite 2>&1 | tee -a "$LOG"
code=${PIPESTATUS[0]}
set -e
log "table_baseline exit=${code}"
[[ $code -ne 0 ]] && exit "$code"

log "alert reduction summary"
"$PY" - "$OUT" <<'PY' | tee -a "$LOG"
import json
import sys

metrics = json.load(open(f"{sys.argv[1]}/metrics.json"))
print("run_id:", metrics["run_id"])
rules = metrics["test_metrics"].get("rules")
if rules:
    budget = rules["alert_budgets"]
    print(f"rule baseline: pr_auc={rules['pr_auc']:.4f}")
    print("rule budgets:", {k: round(v["recall_at_k"], 4) for k, v in budget.items()})
reduction = metrics.get("alert_reduction_vs_rules") or {}
if not reduction:
    print("WARNING alert_reduction_vs_rules is still empty")
for model, payload in reduction.items():
    print(f"-- {model}: {json.dumps(payload, ensure_ascii=False)[:400]}")
PY

log "DONE"
