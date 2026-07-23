# 实施状态（2026-07-23）

## 已实现并有自动化回归的能力

- P0：公开 SAML-D Schema 契约、聚合质量清单、固定时间外切分、数据转换和 PIT 构建
  的运行清单、本地产物隔离、强内部 API 令牌校验。
- P1：严格 PIT 窗口、版本化审批规则与命中证据、规则/Logistic/CatBoost/图统计 +
  CatBoost 基线、稳定哈希或训练期 OOF hard-negative 采样、概率指标、稳定性与新账户
  切片、支付方式/地域组合/币种组合切片、漂移、局部 SHAP、资源与吞吐记录、逐列
  特征登记表，以及固定召回下相对规则基线的告警削减率。
- P2：训练期节点索引、未知账户哈希桶、仅历史边的 GraphSAGE、CPU/GPU 训练、图路径
  证据、时间滚动 OOF、验证期校准和阈值、冻结融合器的独立测试期评估；
  GAT/RGCN/PNA 边分类架构脚手架（配置可选，全量对比待 PIT 后执行）。
- P3：RiskEvidencePackage、YAML Typology + 本地 BM25、LangGraph 单 Agent、SAR 草稿、
  外部注释最小化与事实校验、版本化 Prompt、Golden Set 运行清单、FastAPI、内部鉴权、
  人工复核审计记录和只含虚构数据的 Demo；冻结交易分数的账户风险、资金路径和关联子图
  调查视图由独立无标签聚合器生成。

受控批量评分可选择恢复表格基线、GraphSAGE 和冻结融合产物。图推理使用无标签快照；
调用方无法传入路径或交易 payload，告警 ID 为随机不透明引用。

## 已完成的本地验证

- ruff / pytest / Golden mock；CI 已在 Python 3.11 + PyG 采样后端下全绿。
- Docker Compose Demo 烟雾：`/healthz`、`/demo` 返回 200。
- SAML-D 全量转换已完成；全量 PIT 特征构建进行中（与烟雾路径并行，互不覆盖）。
- 烟雾路径已跑通完整工程链路：`prepared_smoke`（12 日，含 3/4 月训练日）→
  `pit_features_smoke` → `models_smoke/{table,graphsage,table_oof,graph_oof,fusion,fusion_test,investigation_views}`
  （`models.smoke.yaml`；OOF 使用 `--splits 1 --minimum-training-months 1`）。仅链路验证。
- Demo 页已强化「虚构 / 非指标」边界说明，并展示 SAR 草稿摘要；Golden Mock 附
  `golden/README.md` 边界说明。
- 全量 PIT 完成后的正式命令清单：`docs/FULL_RUN_AFTER_PIT.md`。
- ECNU ecnu-max 曾对虚构 Golden case 做过调查注释 smoke test（非模型效果）。

## 尚不能声称完成的实证工作

- 全量 PIT → 训练 → 融合 → 冻结测试尚未完成；因此没有任何可对外引用的效果指标。
  烟雾集上的 PR-AUC / 告警预算数字仅作调试参考，不得写入模型卡或对外材料。
  全量 OOF 须用默认 `--splits 3 --minimum-training-months 2`，勿沿用烟雾参数。
- 约 100 个经人工复核的 Golden cases 与正式晋升阈值仍待补齐；Mock cases 不能替代。

## 验收命令

    D:\Miniconda3\envs\aml-evidence\python.exe -m ruff check src tests
    D:\Miniconda3\envs\aml-evidence\python.exe -m pytest
    D:\Miniconda3\envs\aml-evidence\Scripts\aml-evaluate-golden.exe --cases golden/mock_cases.json --typologies knowledge/typologies --output artifacts/golden_summary.json

真实数据运行前后均应保留 artifacts 中对应的 run manifest、指标报告、模型版本、
校准阈值和 Prompt/Typology 版本；不要把其中的本地完整产物复制到公开仓库。
