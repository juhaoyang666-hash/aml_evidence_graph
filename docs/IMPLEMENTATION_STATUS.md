# 实施状态（2026-07-25 全量链路完成）

> **2026-07-25**：全量 PIT → table_baseline(_rules) → GraphSAGE → table_oof →
> graph_oof → fusion → fusion_test → investigation_views 已完成
> （`logs/remaining_chain_done.flag`）。主线指标见 [RESULTS.md](RESULTS.md)；
> 模型卡已回填 [MODEL_CARD.md](MODEL_CARD.md)。规划修订见
> [PLAN_REVISION.md](PLAN_REVISION.md)。规则基线见 [RULE_BASELINE.md](RULE_BASELINE.md)。

## 已实现并有自动化回归的能力

- P0：公开 SAML-D Schema 契约、聚合质量清单、固定时间外切分、数据转换和 PIT 构建
  的运行清单、本地产物隔离、强内部 API 令牌校验。
- P1：严格 PIT 窗口、版本化审批规则与命中证据、规则/Logistic/CatBoost/图统计 +
  CatBoost 基线、稳定哈希或训练期 OOF hard-negative 采样、概率指标、稳定性与新账户
  切片、支付方式/地域组合/币种组合切片、漂移、局部 SHAP、资源与吞吐记录、逐列
  特征登记表，以及固定召回下相对规则基线的告警削减率。
- P2：训练期节点索引、未知账户哈希桶、仅历史边的 GraphSAGE、CPU/GPU 训练、图路径
  证据、时间滚动 OOF、验证期校准和阈值、冻结融合器的独立测试期评估；
  GAT/RGCN/PNA 边分类架构（配置可选；同协议全量对比已完成，见 RESULTS）。
- P3：RiskEvidencePackage、YAML Typology + 本地 BM25、LangGraph 单 Agent、SAR 草稿、
  外部注释最小化与事实校验、版本化 Prompt、Golden Set 运行清单、FastAPI、内部鉴权、
  人工复核审计记录和只含虚构数据的 Demo；冻结交易分数的账户风险、资金路径和关联子图
  调查视图由独立无标签聚合器生成。

受控批量评分可选择恢复表格基线、GraphSAGE 和冻结融合产物。图推理使用无标签快照；
调用方无法传入路径或交易 payload，告警 ID 为随机不透明引用。

## 全量本地产物（摘要）

| 阶段 | 产物目录 | 状态 |
|---|---|---|
| PIT | `artifacts/pit_features` | 完成（321 分区 / 9,504,852 行） |
| 表格基线 + 规则 KPI | `artifacts/table_baseline_rules` | 完成；CatBoost 测试 PR-AUC **0.8092** |
| GraphSAGE | `artifacts/graphsage` | 完成；测试 PR-AUC **0.8777** |
| table_oof / graph_oof | `artifacts/table_oof`, `artifacts/graph_oof` | 完成（默认 3 splits / min 2 months） |
| 三路融合（含 graph_stats） | `artifacts/fusion`, `artifacts/fusion_test` | 完成；作对照 |
| 双路融合 catboost+graphsage | `artifacts/fusion_cb_gs`, `artifacts/fusion_test_cb_gs` | 完成；测试 PR-AUC **0.8973** |
| 调查视图 | `artifacts/test_investigation_views` | 完成 |
| Golden 30 | `golden/cases_v1.json` + `golden/adjudication_v1.json` | 用户授权 agent 裁定完成（非第三方面板） |
| GAT/RGCN/PNA | `artifacts/{gat,rgcn,pna}` | **完成**；测试 PR-AUC GAT **0.9483** / RGCN **0.9031** / PNA **0.7049**（vs GraphSAGE 0.8777） |

## 已完成的本地验证

- ruff / pytest / Golden mock；CI 已在 Python 3.11 + PyG 采样后端下全绿。
- Docker Compose Demo 烟雾：`/healthz`、`/demo` 返回 200。
- SAML-D 全量转换与全量 PIT 特征构建已完成。
- 烟雾路径已跑通完整工程链路：`prepared_smoke` → `pit_features_smoke` →
  `models_smoke/{table,graphsage,table_oof,graph_oof,fusion,fusion_test,investigation_views}`
  （仅链路验证，不得当作效果数字）。
- Demo 页已强化「虚构 / 非指标」边界说明；Golden 边界见 `golden/README.md`。
- 全量复现命令：`docs/FULL_RUN_AFTER_PIT.md`；剩余 GPU 链：`scripts/run_remaining_gpu.sh`。
- ECNU ecnu-max 可用于调查注释；默认 `AML_LLM_ENABLED` 受控，不影响评分链路。

## 尚待 / 进行中

- **GAT / RGCN / PNA 全量对比**：~~进行中~~ **已完成**；数字见 [RESULTS.md](RESULTS.md)
  「Edge GNN architecture comparison」。主线仍报 CatBoost / GraphSAGE / 双路融合。
- **Golden 裁定**：`adjudication_v1.json` 已完成（adjudicator=`agent-authorized-by-user`）。
  这是项目裁定的回归真相集，**不是**独立第三方人工评审团。
- **PIT 重写 / 大规模架构变更**：无必要则不做。
- 晋升阈值的组织侧审批与真实业务外推：合成基准不能替代。

## 验收命令

```bash
export PYTHONPATH=src
/data1/yangjuhao/envs/risk/bin/python -m ruff check src tests
/data1/yangjuhao/envs/risk/bin/python -m pytest
/data1/yangjuhao/envs/risk/bin/python -m aml_evidence_graph.investigation.golden \
  --cases golden/cases_v1.json --typologies knowledge/typologies \
  --output artifacts/golden_summary.json
# 可选 LLM：
#   --use-llm   # 需 .env 中 AML_LLM_ENABLED 与 API key
```

真实数据运行前后均应保留 artifacts 中对应的 run manifest、指标报告、模型版本、
校准阈值和 Prompt/Typology 版本；不要把其中的本地完整产物复制到公开仓库。
