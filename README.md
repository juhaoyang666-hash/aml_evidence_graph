# AML Evidence Graph

基于公开合成数据集 **[SAML-D](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)**
（约 950 万笔交易）的交易级 AML 风险识别与调查辅助系统。

- **打分**：PIT 表格特征 + 历史图边分类 + 训练期 OOF 堆叠融合（验证期校准 / 锁定阈值）
- **调查**：Evidence Package 约束下的 Typology 检索、调查注释与 SAR 草稿；LLM **不参与**
  风险分数、阈值或案件结论
- **边界**：固定时间外切分；特征与图邻居只读预测时点之前的历史

完整指标、run_id 与复现说明见 **[docs/RESULTS.md](docs/RESULTS.md)** ·
**[docs/MODEL_CARD.md](docs/MODEL_CARD.md)**。

## 全量主线结果（公开合成 SAML-D，时间外测试）

| 模型 | 角色 | 测试 PR-AUC | 0.1% 告警预算 P / R |
|---|---|---:|---|
| CatBoost | 主线表格 | **0.809** | 0.858 / 0.737 |
| **GAT** | **主线图** | **0.948** | 0.974 / 0.838 |
| **catboost + GAT** | **主线融合** | **0.918** | 0.936 / 0.805 |
| GraphSAGE / catboost+GS | 原主线（保留可比） | 0.878 / 0.897 | 见 RESULTS |

同召回相对规则基线：CatBoost @50% 召回告警量削减约 **99.9%**（见 RESULTS）。

**须同时披露**：融合（0.918）优于表格与原 GraphSAGE 融合，但**不**优于单独 GAT（0.948）。
`graph_stats` / 三路融合仅作对照，不进主线。

同协议边 GNN：GAT 0.948 > RGCN 0.903 > GraphSAGE 0.878 > PNA 0.705。

## 文档入口

| 文档 | 内容 |
|---|---|
| [RESULTS.md](docs/RESULTS.md) | 全量指标表、融合对比、架构对比、Golden、调查视图 |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | 模型边界、评价协议、限制 |
| [FULL_RUN_AFTER_PIT.md](docs/FULL_RUN_AFTER_PIT.md) | **Linux 全量复现命令**（主线 GAT） |
| [ENVIRONMENT.md](docs/ENVIRONMENT.md) | conda `risk`、GPU、ECNU LLM |
| [RULE_BASELINE.md](docs/RULE_BASELINE.md) | 规则 v2026.2 定阈说明 |
| [LLM_PLAN.md](docs/LLM_PLAN.md) | LLM 职责边界与评测 |
| [IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) | 工程完成度 |

## 环境（Linux，全量实验）

```bash
export PY=/data1/yangjuhao/envs/risk/bin/python   # Python 3.12 · torch 2.5.1+cu121
cd /data1/yangjuhao/反洗钱/aml_evidence_graph
export PYTHONPATH=src
```

- GPU：最多 4× RTX 3090（`--max-gpus` / `CUDA_VISIBLE_DEVICES`）
- 长任务请放在 **tmux** 中
- 详细约定：[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)

## 安全与数据边界

- 对外须标注为**公开合成基准**，不得包装成私有/真实业务数据
- 完整交易、模型与报告在 `artifacts/`（已 gitignore），仓库默认仅 Mock / Schema
- 受控 API 只接受日期分区与告警引用；启用受控模式须配置强内部令牌

## 快速烟雾（工程验证，指标不可对外引用）

```bash
export PYTHONPATH=src
$PY -m aml_evidence_graph.ingestion.smoke_subset \
  --input artifacts/prepared_transactions --output artifacts/prepared_smoke --overwrite
$PY -m aml_evidence_graph.features.build \
  --input artifacts/prepared_smoke --output artifacts/pit_features_smoke --overwrite
# 后续 table / graphsage / OOF / fusion 见 docs/FULL_RUN_AFTER_PIT.md 烟雾对照表
# 或沿用 configs/models.smoke.yaml + artifacts/models_smoke/
```

全量 OOF 使用默认 `--splits 3 --minimum-training-months 2`。  
**主线全量复现（GAT + catboost+GAT）**：[docs/FULL_RUN_AFTER_PIT.md](docs/FULL_RUN_AFTER_PIT.md)。

辅助脚本：`scripts/run_full_train_chain.sh`、`scripts/run_remaining_gpu.sh`、
`scripts/run_arch_comparison.sh`、`scripts/backfill_rule_hits.py`。

## 调查视图 / API / Golden

冻结融合分后生成账户风险、资金路径与案件视图（主线目录
`artifacts/test_investigation_views_gat`），命令见 FULL_RUN 文档。

```bash
# Mock Demo（无完整交易）
$PY -m aml_evidence_graph.api.app   # 或安装后的 aml-api；浏览器 http://127.0.0.1:8000/demo

# Golden（30 案裁定集；默认不调外部 LLM）
$PY -m aml_evidence_graph.investigation.golden \
  --cases golden/cases_v1.json --typologies knowledge/typologies \
  --output artifacts/golden_summary.json
# 可选：--use-llm（需 AML_LLM_ENABLED 与 ECNU_API_KEY，见 ENVIRONMENT.md）
```

## 本地验收

```bash
export PYTHONPATH=src
$PY -m ruff check src tests
$PY -m pytest
docker compose -f docker-compose.demo.yml up --build
```
