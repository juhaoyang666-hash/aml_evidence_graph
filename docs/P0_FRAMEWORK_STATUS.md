# 求职改进 P0 框架状态

本页记录 `RISK_ALGORITHM_INTERNSHIP_PROJECT_IMPROVEMENT_2026.md` 中 P0 建议的代码落地状态。
框架可运行不等于正式指标已完成；只有扩容后的 Golden、完整性能报告和完成产物才能写入简历。

## 已落地

| 能力 | 实现 | 当前边界 |
|---|---|---|
| 混合 Typology 检索 | BM25 + TF-IDF dense + RRF + lexical-overlap rerank | TF-IDF baseline 不是语义大模型 embedding |
| 检索评测 | Recall@1/3、MRR、nDCG@3、无答案误召回 | 当前只有 15 条框架种子集，目标至少 80 条 |
| 只读调查工具 | PIT 特征快照、有界历史子图、Typology 检索 | 当前从冻结 EvidenceStore 读取，不接受 SQL/路径/URL |
| 工具路由 | 确定性、最多 4 次的结构化请求 | 尚未启用 LLM 自主路由 |
| HITL | LangGraph interrupt + approve/edit/reject + checkpoint | 内存 checkpoint 已测试；SQLite 为可选环境 |
| 审计 | 工具名、参数名、输出键、耗时、状态与错误码进入 checkpoint state | 当前不落完整值；生产仍需追加独立审计存储和保留策略 |
| 简历证据 | 只读取 metrics + manifest，并拒绝缺失或运行中的必需产物 | 还需给 manifest 增加正式/Smoke 标识后才能自动拒绝 Smoke |
| API 压测 | 异步 HTTP 客户端，输出吞吐、错误率和 p50/p95/p99 | 本机结果不是生产 SLA |

## 轻量验证命令

```powershell
$PY = "D:\Miniconda3\envs\aml-evidence\python.exe"
$env:PYTHONPATH = "src"

& $PY scripts/evaluate_retrieval.py
& $PY scripts/build_resume_evidence.py --allow-incomplete `
  --markdown artifacts/resume_evidence/RESUME_EVIDENCE_PREVIEW.md
& $PY scripts/benchmark_api.py --base-url http://127.0.0.1:8000 `
  --path /healthz --method GET
& $PY scripts/check_career_environment.py
& $PY -m pytest tests/test_retrieval.py tests/test_controlled_agent.py `
  tests/test_resume_evidence.py tests/test_serving_benchmark.py
```

以上命令不启动模型训练，也不需要 GPU。

## 可选环境

使用清华 PyPI 镜像按目标任务安装；不要一次安装所有重型能力：

```powershell
$env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

# SQLite 持久化 checkpoint，已在当前 aml-evidence 环境安装并完成重启恢复烟雾验证
& $PY -m pip install -e ".[llm,agent-persistence]"

# 需要真正语义 embedding 时再装；模型权重不会随包安装自动进入仓库
& $PY -m pip install -e ".[retrieval]"

# 以下属于 P1，按目标 JD 选择
& $PY -m pip install -e ".[mlops]"
& $PY -m pip install -e ".[spark]"
```

`benchmark` 仅依赖 httpx，已包含在当前 LLM 环境中。当前额外安装的是
`langgraph-checkpoint-sqlite 3.1.0`；SentenceTransformer、MLflow 和 PySpark 均未在后台
GAT 实验期间安装，避免额外内存、磁盘和依赖扰动。当前 PyTorch 为 GPU 版
`2.5.1+cu121` 且 CUDA 可用；Java 未安装，因此 Spark 骨架暂不运行。也可使用
`scripts/install_career_extras.ps1 -Group <agent|retrieval|mlops|spark>` 通过清华镜像分组安装。

## 2026-07-28 本机轻量验收

- 本次框架定向测试：13 passed；全仓测试：90 passed；新增文件 Ruff：通过。
- SQLite checkpoint：关闭连接、重新创建工作流后，同一 `thread_id` 审批恢复成功。
- Mock `/healthz`：100 请求、并发 5、错误率 0%，p95 13.60 ms。
- Mock 调查草稿接口：50 请求、并发 5、错误率 0%，p95 123.48 ms。
- 检索种子集：15 案；本次 TF-IDF 的 hybrid 未稳定优于 BM25，因此尚未切换默认检索器。
- 简历证据预览：仅识别到 FE v2 CatBoost；主线三项产物在本机缺失，GAT/融合仍在运行，
  `public_ready=false`。不得把预览当作最终简历证据。

压测产物位于 `artifacts/serving_benchmark_*_p0*`，属于当前 Windows 本机观测，不是生产 SLA。

## P1 代码入口（未完成环境验收）

- `features/spark_replay.py`：五个代表性 PIT 窗口特征，窗口严格截止 `t-1`，不使用普通
  笛卡尔积；待 Java/PySpark 环境就绪后验证等价率、shuffle、耗时和峰值资源。
- `tracking/mlflow_adapter.py`：保留现有 manifest 为权威记录，只同步聚合指标；candidate
  gate 明确拒绝测试集指标。待 FE v2 完成且按需安装 MLflow 后接入实际产物。

## 尚未完成

1. 将检索集扩展到至少 80 条，并完成 BM25/dense/hybrid/rerank 正式对照。
2. 在 API 中暴露 thread_id、暂停、恢复和审批接口；当前 v2 先作为库内可测试工作流。
3. 为工具路由构建至少 60 案 Agent Golden 和 Bad Case 分类。
4. 对 Mock、冻结模型评分与 Agent 三条路径分别压测并生成 `SERVING_BENCHMARK.md`。
5. FE v2 实验完成后生成唯一 `RESUME_EVIDENCE.md`，再更新公开简历数字。
6. 为 run manifest 增加正式/Smoke 明确标识，并让简历证据构建器强制校验该字段。
