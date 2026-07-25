#!/usr/bin/env bash
set -euo pipefail
cd /data1/yangjuhao/反洗钱/aml_evidence_graph
mkdir -p logs
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
unset PYTHONPATH PYTHONUSERBASE PIP_USER || true
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR=/data1/yangjuhao/pip-cache
PY=/data1/yangjuhao/envs/risk/bin/python
PIP=/data1/yangjuhao/envs/risk/bin/pip
LOG=logs/pip_risk_full.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] BEGIN pip into risk site-packages"
echo "ns=$(readlink /proc/$$/ns/net)"
$PY -V
$PY -c 'import sys; print("prefix", sys.prefix)'
$PY -c 'import sysconfig,os; p=sysconfig.get_path("purelib"); print(p, "writable", os.access(p, os.W_OK))'
touch /data1/yangjuhao/envs/risk/lib/python3.12/site-packages/.wtest
rm /data1/yangjuhao/envs/risk/lib/python3.12/site-packages/.wtest
echo WRITE_OK

# Proxy + official indexes only (no Tsinghua)
curl -sI --connect-timeout 5 -x http://127.0.0.1:7890 https://pypi.org/simple/ | head -2

echo "[$(date -Is)] install torch cu121"
$PIP install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[$(date -Is)] verify torch in risk"
$PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "file", torch.__file__)'

echo "[$(date -Is)] install torch_geometric"
$PIP install torch_geometric -i https://pypi.org/simple

echo "[$(date -Is)] install pyg extensions (best-effort)"
set +e
$PIP install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html \
  -i https://pypi.org/simple
EXT_RC=$?
set -e
if [[ $EXT_RC -ne 0 ]]; then
  echo "WARN: pyg extensions exit=$EXT_RC (non-fatal for GraphSAGE)"
fi

echo "[$(date -Is)] final verify"
$PY -c 'import torch, torch_geometric; from torch_geometric.nn import SAGEConv; print("OK", torch.__version__, torch_geometric.__version__, torch.__file__)'
ls /data1/yangjuhao/envs/risk/lib/python3.12/site-packages | grep -iE '^torch' | head -20
echo "[$(date -Is)] DONE_PIP_RISK"
date -Is > logs/pip_risk_done.flag
