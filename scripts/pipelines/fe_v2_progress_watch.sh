#!/usr/bin/env bash
set -u
ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
cd "$ROOT" || exit 1
LOG="$ROOT/logs/fe_v2_progress_watch.log"
STATUS="$ROOT/artifacts/logs/fe_v2_pipeline_status.json"
METRICS="$ROOT/artifacts/table_baseline_fe_v2/metrics.json"
PIPELOG="$ROOT/logs/fe_v2_pipeline.log"
BASELINE=0.8092
mkdir -p logs

check_once() {
  local now
  now=$(date '+%Y-%m-%d %H:%M:%S')
  {
    echo "======== 进度检查 ${now} ========"
    python3 - <<'PY'
import json, os, glob, time
from datetime import datetime
root="/data1/yangjuhao/反洗钱/aml_evidence_graph"
os.chdir(root)
status={}
sp="artifacts/logs/fe_v2_pipeline_status.json"
if os.path.exists(sp):
    status=json.load(open(sp))
s1=status.get("steps",{}).get("01_pit_fe_v2",{})
s2=status.get("steps",{}).get("02_table_fe_v2",{})
print(f"阶段: 01_pit={s1.get('status')} | 02_table={s2.get('status')} complete={s2.get('complete')} error={s2.get('error')}")
# process
alive=False
pids=[]
try:
    import subprocess
    out=subprocess.check_output(["bash","-lc","ps aux | grep '[r]un_fe_v2_pipeline_resumable' || true"], text=True)
    for line in out.strip().splitlines():
        if line.strip():
            parts=line.split()
            pids.append(parts[1])
            alive=True
            print(f"进程: pid={parts[1]} cpu={parts[2]}% mem={parts[3]}% etime≈rss={parts[5]} cmd={' '.join(parts[10:14])}")
except Exception as e:
    print("进程查询失败", e)
if not alive:
    print("进程: run_fe_v2 未在运行")
# PIT
pit=None
for c in ["artifacts/pit_features_fe_v2","data/pit_features_fe_v2"]:
    if os.path.isdir(c):
        pit=c; break
summary={}
if pit and os.path.exists(os.path.join(pit,"_feature_build_summary.json")):
    summary=json.load(open(os.path.join(pit,"_feature_build_summary.json")))
elif s1.get("summary"):
    summary=s1["summary"]
dirs=[]
if pit:
    dirs=sorted([d for d in os.listdir(pit) if os.path.isdir(os.path.join(pit,d)) and not d.startswith("_")])
if dirs:
    print(f"PIT: 路径={pit} 分区数={len(dirs)} 日期范围={dirs[0]} ~ {dirs[-1]}")
elif summary:
    print(f"PIT: 分区={summary.get('partition_count')} 日期={summary.get('event_date_min')}~{summary.get('event_date_max')}")
else:
    print("PIT: 缺失")
if summary:
    print(f"关键/规模: rows={summary.get('row_count')} rule_hit={summary.get('rule_hit_count')} feat_cols={summary.get('feature_column_count')}")
# metrics
mp="artifacts/table_baseline_fe_v2/metrics.json"
if os.path.exists(mp):
    m=json.load(open(mp))
    # flexible extract
    pra=None
    for path in [
        ("test","pr_auc"),("test_metrics","pr_auc"),("metrics","test","pr_auc")
    ]:
        cur=m
        ok=True
        for k in path:
            if isinstance(cur,dict) and k in cur:
                cur=cur[k]
            else:
                ok=False; break
        if ok and isinstance(cur,(int,float)):
            pra=cur; break
    if pra is None:
        pra=m.get("pr_auc") or m.get("test_pr_auc") or m.get("PR-AUC")
    print(f"PR-AUC: {pra} (基线 0.8092, delta={None if pra is None else round(float(pra)-0.8092,6)})")
    print("metrics_json:", json.dumps(m, ensure_ascii=False)[:800])
else:
    print("PR-AUC: 尚未产出")
# log tail
pl="logs/fe_v2_pipeline.log"
if os.path.exists(pl):
    print("--- log tail ---")
    try:
        with open(pl,"rb") as f:
            f.seek(0,2); size=f.tell(); f.seek(max(0,size-2500))
            print(f.read().decode("utf-8","replace")[-2000:])
    except Exception as e:
        print("log read err", e)
# table artifacts growth
tb="artifacts/table_baseline_fe_v2"
if os.path.isdir(tb):
    files=sorted(os.listdir(tb))
    print(f"table_baseline_fe_v2 文件: {files[:30]}")
print("")
# done?
done=False
if s2.get("complete") or s2.get("status") in ("complete","failed","error"):
    done=True
if os.path.exists(mp) and (s2.get("complete") or s2.get("status")=="complete"):
    done=True
# exit code in log
if os.path.exists(pl):
    txt=open(pl,encoding="utf-8",errors="replace").read()
    if "EXIT_CODE=" in txt:
        done=True
print(f"DONE_FLAG={1 if done else 0} ALIVE={1 if alive else 0}")
PY
  } >> "$LOG" 2>&1
}

# first immediate check if called with --once
if [[ "${1:-}" == "--once" ]]; then
  check_once
  exit 0
fi

while true; do
  check_once
  # stop if done
  if python3 - <<'PY'
import json, os
sp="artifacts/logs/fe_v2_pipeline_status.json"
mp="artifacts/table_baseline_fe_v2/metrics.json"
pl="logs/fe_v2_pipeline.log"
done=False
if os.path.exists(sp):
    d=json.load(open(sp))
    s2=d.get("steps",{}).get("02_table_fe_v2",{})
    if s2.get("complete") or s2.get("status") in ("complete","failed","error"):
        done=True
if os.path.exists(pl) and "EXIT_CODE=" in open(pl,encoding="utf-8",errors="replace").read():
    done=True
# also stop if process dead and not complete after grace - still report
import subprocess
alive=subprocess.call(["bash","-lc","pgrep -f 'run_fe_v2_pipeline_resumable.py' >/dev/null"])==0
if (not alive) and os.path.exists(mp):
    done=True
if (not alive) and os.path.exists(sp):
    d=json.load(open(sp)); s2=d.get("steps",{}).get("02_table_fe_v2",{})
    if s2.get("status") in ("failed","error","complete") or s2.get("complete"):
        done=True
    elif not alive:
        # dead without success
        done=True
raise SystemExit(0 if done else 1)
PY
  then
    echo "$(date -Is) watch loop exiting (pipeline finished or dead)" >> "$LOG"
    break
  fi
  sleep 600
done
