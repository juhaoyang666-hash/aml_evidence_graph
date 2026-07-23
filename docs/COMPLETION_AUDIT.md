# AML Evidence Graph 完成度审计

审计日期：2026-07-23。范围是本目录中新建的 `aml_evidence_graph`；原有项目材料未被修改。
“已实现”表示源码、自动化测试和本机环境验证已具备，不代表已在 SAML-D 全量数据上得到业务指标。

| 计划要求 | 状态 | 可追溯实现/证据 |
|---|---|---|
| 交易标签作为边分类目标，账户/案件不直接使用标签 | 已实现 | `data/contract.py`、`models/graphsage.py`；`aggregation/views.py` 显式拒绝标签列。 |
| 固定时间外切分与 PIT 防泄漏 | 已实现 | `data/splits.py`、`features/pit.py`、`graph/snapshots.py`，含同秒批次隔离测试。 |
| 公开 SAML-D CSV 到分区 Parquet、质量清单与运行清单 | 已实现；全量转换已跑通，PIT/训练待完成 | `data/configuration.py`；`ingestion/profile.py`、`ingestion/prepare.py`、`tracking/run.py`。 |
| 版本化规则、B0/B1/B2/B3、训练期 OOF hard negative、告警削减 KPI | 已实现，待全量执行 | `rules/engine.py`、`training/table_baseline.py`、`training/oof.py`、`evaluation/metrics.py`。 |
| 历史图 GraphSAGE + 多架构脚手架、无标签推理、融合/校准 | GraphSAGE 已实现；GAT/RGCN/PNA 脚手架已落地，待全量对比 | `graph/snapshots.py`、`models/edge_classifiers.py`、`training/graphsage.py`、`training/fusion.py`。 |
| 排序、校准、告警预算、切片、资源和漂移评估 | 已实现，待全量执行 | `evaluation/metrics.py`、`evaluation/monitoring.py`、`evaluation/drift.py`。 |
| 特征的版本、负责人、来源、窗口与单测登记 | 已实现 | `configs/features.yaml`、`features/registry.py`。 |
| 账户风险、资金路径、关联子图案件视图 | 已实现，待冻结分数执行 | `aggregation/views.py`。 |
| Evidence Package、Typology BM25、单 Agent、SAR 草稿、事实校验 | 已实现 | `evidence/`、`investigation/`、`configs/prompts/`。 |
| Golden 评测、Prompt/用量/成本记录 | 框架与扩展 Mock cases 已实现 | `investigation/golden.py`、`golden/mock_cases.json`；正式 100 案人工标注仍待补。 |
| 内部 API、人工复核、Mock Demo、Docker、CI | 已实现并通过 | `api/`、`Dockerfile`、`docker-compose.demo.yml`、`.github/workflows/ci.yml`（`e50ce5f` 全绿）。 |

## 本机可复核结果

- `python -m ruff check src tests` / `python -m pytest`：以当前仓库测试为准。
- Docker Demo：`/healthz` 与 `/demo` 已烟雾通过。
- CI：[run 29982602058](https://github.com/juhaoyang666-hash/aml_evidence_graph/actions/runs/29982602058) success。
- SAML-D 全量转换：`run_id=20260723T040559Z-02a632c9c2`；PIT 全量构建进行中。
- 烟雾全链路：`pit_features_smoke`（`20260723T063912Z-27a5e63206`）→
  `models_smoke/` 下 table / GraphSAGE / OOF / fusion / fusion_test / investigation_views。
  指标不可对外引用；全量须换路径与默认 OOF 参数重跑。
- 正式复现清单：`docs/FULL_RUN_AFTER_PIT.md`。

## 仍需等待或外部输入的验收项

1. 全量 PIT 完成后的表格/图训练、融合与冻结测试评估；完成后才可填写真实指标与告警削减率。
2. 扩展至约 100 个经复核的 Golden cases、调查容量预算、模型晋升阈值。
3. 成熟版：在同一协议下完成 GAT/RGCN/PNA 全量对比实验。
