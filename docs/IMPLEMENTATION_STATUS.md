# 实施状态（2026-07-22）

## 已实现并有自动化回归的能力

- P0：私有 CSV Schema 契约、聚合质量清单、固定时间外切分、HMAC 账户 token 化、
  数据转换和 PIT 构建的运行清单、私有产物隔离、强密钥校验。
- P1：严格 PIT 窗口、版本化审批规则与命中证据、规则/Logistic/CatBoost/图统计 +
  CatBoost 基线、稳定哈希或训练期 OOF hard-negative 采样、概率指标、稳定性与新账户
  切片、支付方式/地域组合/币种组合切片、漂移、局部 SHAP、资源与吞吐记录，以及逐列
  特征登记表。
- P2：训练期节点索引、未知账户哈希桶、仅历史边的 GraphSAGE、CPU/GPU 训练、图路径
  证据、时间滚动 OOF、验证期校准和阈值、冻结融合器的独立测试期评估。
- P3：RiskEvidencePackage、YAML Typology + 本地 BM25、LangGraph 单 Agent、外部
  注释最小化与事实校验、版本化 Prompt、Golden Set 运行清单、FastAPI、内部鉴权、
  人工复核审计记录和只含虚构数据的 Demo；冻结交易分数的账户风险、资金路径和关联子图
  调查视图由独立无标签聚合器生成。

私有批量评分可选择恢复表格基线、GraphSAGE 和冻结融合产物。图推理使用无标签快照；
调用方无法传入路径或交易 payload，告警 ID 为随机不透明引用。

## 已完成的本地验证

- ruff 静态检查与完整 pytest 套件均在 aml-evidence 环境执行（59 passed）。
- CUDA GraphSAGE 冒烟已在 NVIDIA GeForce RTX 2060 上执行过；图模型单元与合成端到端
  测试同时覆盖 CPU 路径。
- ECNU ecnu-max 已只用一个虚构 Golden case 做过调查注释 smoke test：事实快照与
  Schema 校验通过，服务商 token 用量和延迟被写入本地聚合报告。该结果不是私有模型效果。

## 尚不能声称完成的外部/实证工作

- 真实全量转换、训练、融合和测试尚未运行，因为当前会话没有 AML_TOKENIZATION_SECRET；
  因而没有任何可对外引用的真实模型指标或 run_id。
- 100 个经人工双人标注的 Golden cases、调查容量预算和正式模型晋升标准需要业务/合规
  人员提供或确认；仓库仅提供 Mock smoke case 与评测框架，不能把合成案例替代人工标注。
- Docker CLI 在当前主机不可用，因此 Dockerfile 和 compose 配置已静态隔离为 Mock-only，
  但尚未在本机完成镜像构建运行验证。

## 验收命令

    D:\Miniconda3\envs\aml-evidence\python.exe -m ruff check src tests
    D:\Miniconda3\envs\aml-evidence\python.exe -m pytest
    D:\Miniconda3\envs\aml-evidence\Scripts\aml-evaluate-golden.exe --cases golden/mock_cases.json --typologies knowledge/typologies --output artifacts/golden_summary.json

真实数据运行前后均应保留 artifacts 中对应的 run manifest、指标报告、模型版本、
校准阈值和 Prompt/Typology 版本；不要把其中的私有内容复制到公开仓库。
