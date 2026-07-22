# AML Evidence Graph 完成度审计

审计日期：2026-07-22。范围是本目录中新建的 `aml_evidence_graph`；原有项目材料未被修改。
“已实现”表示源码、自动化测试和本机环境验证已具备，不代表已在私有全量数据上得到业务指标。

| 计划要求 | 状态 | 可追溯实现/证据 |
|---|---|---|
| 已确认交易标签作为边分类目标，账户/案件不直接使用标签 | 已实现 | `data/contract.py`、`models/graphsage.py`；`aggregation/views.py` 显式拒绝标签列。 |
| 固定时间外切分与 PIT 防泄漏 | 已实现 | `data/splits.py`、`features/pit.py`、`graph/snapshots.py`，含同秒批次隔离测试。 |
| 已脱敏私有 CSV 到分区 Parquet、质量清单与运行清单 | 已实现，待真实数据执行 | `data/configuration.py` 锁定切分与缺失率门槛；`ingestion/profile.py`、`ingestion/prepare.py`、`tracking/run.py`。 |
| 版本化规则、B0/B1/B2/B3、训练期 OOF hard negative | 已实现，待真实数据执行 | `rules/engine.py`、`training/table_baseline.py`、`training/oof.py`。 |
| 历史图 GraphSAGE、无标签推理、融合/校准 | 已实现，待真实数据执行 | `graph/snapshots.py`、`training/graphsage.py`、`training/fusion.py`；图报告含账户重叠与冷启动聚合。 |
| 排序、校准、告警预算、切片、资源和漂移评估 | 已实现，待真实数据执行 | `evaluation/metrics.py`、`evaluation/monitoring.py`、`evaluation/drift.py`；含月份、模式、新账户、支付方式、地域组合与币种组合切片。 |
| 特征的版本、负责人、来源、窗口与单测登记 | 已实现 | `configs/features.yaml`、`features/registry.py`；PIT 输出写入 `_feature_registry.json`。 |
| 账户风险、资金路径、关联子图案件视图 | 已实现，待冻结分数执行 | `aggregation/views.py`；只从分数/规则证据聚合，路径严格时间递增且有跳数、分支和总量上限。 |
| Evidence Package、Typology BM25、单 Agent、事实校验 | 已实现 | `evidence/`、`investigation/`、`configs/prompts/`。 |
| Golden 评测、Prompt/用量/成本记录 | 框架和 Mock smoke 已实现 | `investigation/golden.py`；一例虚构 ECNU smoke 已完成，不可外推为真实质量指标。 |
| 内部 API、人工复核、Mock Demo、Docker 配置、CI | 已实现；Docker 运行待主机验证 | `api/`、`Dockerfile`、`docker-compose.demo.yml`、`.github/workflows/ci.yml`。 |

## 本机可复核结果

- `python -m ruff check src tests`：通过。
- `python -m pytest`：67 passed（仅含合成/Mock 测试，不读取私有流水）。
- `ecnu-max` 仅对一个虚构 Golden case 做过受约束注释 smoke test；事实快照和 Schema 校验通过。
- `aml-evidence` 环境安装了 CUDA PyTorch；此前已在 RTX 2060 做过 GraphSAGE CUDA smoke。

## 仍需外部输入或环境变化的验收项

1. 尚未对私有全量数据执行转换、训练、冻结融合与一次性测试评估；不得填写真实 PR-AUC、召回、延迟或 run_id。
2. 100 个经双人标注的脱敏 Golden cases、调查容量预算、模型晋升阈值需要业务和合规方批准；Mock case 不能替代该验收。
3. Docker CLI 已安装，但 Docker Desktop 守护进程尚待首次启动和本机镜像构建、启动验证。

在以上三项完成前，最终验收清单中的“真实数据事实”“正式 Golden Set”和“Docker 运行验证”必须保持未勾选。
