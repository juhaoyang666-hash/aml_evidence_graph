# AML Evidence Graph 完成度审计

审计日期：2026-07-23（初版）；**2026-07-26 按 `main` 全量结果回填**。  
范围是本仓库 `aml_evidence_graph`。全量指标以 [RESULTS.md](RESULTS.md) /
[MODEL_CARD.md](MODEL_CARD.md) 为准。

| 计划要求 | 状态 | 可追溯实现/证据 |
|---|---|---|
| 交易标签作为边分类目标，账户/案件不直接使用标签 | 已实现 | `data/contract.py`、`models/graphsage.py`；`aggregation/views.py` 显式拒绝标签列。 |
| 固定时间外切分与 PIT 防泄漏 | 已实现；全量已跑通 | `data/splits.py`、`features/pit.py`、`graph/snapshots.py`。 |
| 公开 SAML-D CSV → 分区 Parquet、质量清单与运行清单 | 已实现；全量完成 | `ingestion/`；`artifacts/prepared_transactions`。 |
| 版本化规则、告警削减 KPI | 已实现；全量 KPI 已产出 | `configs/rules/default.yaml` v2026.2、`docs/RULE_BASELINE.md`、`artifacts/table_baseline_rules`。 |
| 历史图边分类 + 多架构、无标签推理、融合/校准 | **已完成**；主线图为 **GAT**，主线融合为 **catboost+GAT** | `artifacts/{gat,graph_oof_gat,fusion_cb_gat,fusion_test_cb_gat}`；对照见 GraphSAGE / RGCN / PNA。 |
| 图统计 + CatBoost | 已实现；**对照**，不进主线融合 | `features/graph_stats.py`；见 RESULTS。 |
| 排序、校准、告警预算、切片、漂移 | 已实现；全量已执行 | `evaluation/`；各 `metrics.json`。 |
| 特征登记 | 已实现 | `configs/features.yaml`、`features/registry.py`。 |
| 账户风险 / 资金路径 / 案件视图 | 已实现；GAT 冻结分已跑 | `artifacts/test_investigation_views_gat`。 |
| Evidence Package、Typology、单 Agent、SAR、事实校验 | 已实现 | `evidence/`、`investigation/`。 |
| Golden 评测 | **30 案 v1 已裁定**（用户授权 agent，非第三方面板） | `golden/cases_v1.json`、`golden/adjudication_v1.json`。 |
| 内部 API、Demo、Docker、CI | 已实现并通过 | `api/`、CI 工作流。 |

## 本机可复核结果

- `ruff` / `pytest` / Golden（`cases_v1.json`）：以当前仓库为准。
- Docker Demo：`/healthz`、`/demo` 烟雾通过。
- 全量转换 / PIT / 训练 / 融合 / 调查视图：见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。
- 正式数字与 run_id：见 [RESULTS.md](RESULTS.md)。

## FE v2 sidecar 审计补记（2026-07-27）

- 本次 `fe_v2`（Polars + P0/P1）已跑完，状态 complete。
- 指标唯一真值来源：`artifacts/table_baseline_fe_v2/metrics.json`。
- 运行状态时间来源：`artifacts/logs/fe_v2_pipeline_status.json`（仅状态/时间）。
- sidecar 结果：validation PR-AUC **0.8660898669**、validation ROC-AUC **0.9982223689**、
  test PR-AUC **0.8754139061**、test ROC-AUC **0.9984022445**。
- 对主线 CatBoost test PR-AUC 0.8092 的差值为 **+0.0662**（四舍五入）。
- 口径约束：该结果仅为 sidecar 路线补充（sidecar 不覆盖主线），不替代主线；smoke 产物只做链路验证，不作为
  对外指标。
- 风险提示：仍需后续复核是否存在过拟合、采样差异或数据切分差异。

## 组织侧仍开放（非工程缺口）

1. 合成基准不能替代真实业务外推与晋升阈值审批。
2. 若需独立第三方人工评审团，可在现有 30 案之上另做外审（当前 v1 为项目内裁定）。
