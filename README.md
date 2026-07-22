# AML Evidence Graph

面向已确认案件标签的交易级 AML 风险识别与调查辅助系统。原有的
反洗钱案例展示目录未被修改，也不是本项目的运行依赖。

主任务是以 is_laundering 作为交易边二分类标签；账户风险、资金路径与案件视图只从
交易风险和历史证据聚合得到。LLM 只生成受 Evidence Package 约束的调查注释，不参与
风险分数、阈值或案件结论。

## 安全与数据边界

- 正式训练、验证和测试仅使用本机私有完整数据；仓库和 Demo 只含 Mock 数据、Schema 与聚合产物。
- 账户写入 Parquet 前会 HMAC token 化。真实密钥、完整交易、模型和私有报告均在 Git 之外。
- 训练、验证、测试使用固定时间外切分；特征和图邻居只读取预测时点之前的数据。
- 私有 API 只接受日期分区和告警引用，不接受调用方上传的交易记录。启用私有模式必须配置强内部令牌。

当前没有执行真实全量训练，因此不得引用任何真实 PR-AUC、召回或业务提升指标。
模型使用边界、评价协议和已知限制见 [模型卡](docs/MODEL_CARD.md)。

## 环境

环境名为 aml-evidence。普通依赖使用清华 PyPI 镜像安装；Windows CUDA PyTorch 使用官方
cu121 轮子，因为镜像不提供对应 GPU 轮子。项目路径包含中文时，请使用普通安装而非
editable 安装：

    D:\Miniconda3\envs\aml-evidence\python.exe -m pip install --no-deps .

完整的密钥、私有产物和 ECNU 调查注释约定见 docs/ENVIRONMENT.md。

## 私有训练与冻结评估

先生成只含聚合信息的数据清单：

    D:\Miniconda3\envs\aml-evidence\Scripts\aml-profile-data.exe --input ../data/SAML-D.csv --output artifacts/data_manifest.json

仅在当前 PowerShell 会话从密钥管理工具注入至少 32 个字符的随机令牌化密钥：

    $env:AML_TOKENIZATION_SECRET = "<由密钥管理工具提供>"

随后按以下顺序运行。所有路径均应位于本机 artifacts 目录；不要提交其内容。

    D:\Miniconda3\envs\aml-evidence\Scripts\aml-convert-private-data.exe --input ../data/SAML-D.csv --output artifacts/tokenized_transactions
    D:\Miniconda3\envs\aml-evidence\Scripts\aml-build-pit-features.exe --input artifacts/tokenized_transactions --output artifacts/pit_features --rules configs/rules/default.yaml
    D:\Miniconda3\envs\aml-evidence\Scripts\aml-train-table.exe --features artifacts/pit_features --output artifacts/table_baseline
    D:\Miniconda3\envs\aml-evidence\Scripts\aml-train-graphsage.exe --features artifacts/pit_features --output artifacts/graphsage
    D:\Miniconda3\envs\aml-evidence\Scripts\aml-generate-table-oof.exe --features artifacts/pit_features --output artifacts/table_oof --model table
    D:\Miniconda3\envs\aml-evidence\Scripts\aml-generate-table-oof.exe --features artifacts/pit_features --output artifacts/graph_oof --model graphsage

融合器只能看训练期 expanding-time OOF 分数；校准器和告警阈值只能看验证期：

    D:\Miniconda3\envs\aml-evidence\Scripts\aml-fit-fusion.exe --oof artifacts/table_oof/table_oof_scores.parquet --oof artifacts/graph_oof/graphsage_oof_scores.parquet --validation artifacts/table_baseline/scores/table_validation_scores.parquet --validation artifacts/graphsage/scores/graphsage_validation_scores.parquet --components catboost,graph_stats_catboost,graphsage --output artifacts/fusion

冻结后才对测试期作一次性评估；此命令不接收训练或验证输入，也不会重新选阈值：

    D:\Miniconda3\envs\aml-evidence\Scripts\aml-evaluate-fusion.exe --fusion-dir artifacts/fusion --test artifacts/table_baseline/scores/table_test_scores.parquet --test artifacts/graphsage/scores/graphsage_test_scores.parquet --features artifacts/pit_features --output artifacts/fusion_test

训练、OOF 和融合命令默认读取 configs/models.yaml；可以用 --model-config 指向经审查的
新版本。实际文件哈希会写入对应 run manifest，CLI 的随机种子覆盖配置中的同名默认值。
PIT 构建同时读取 configs/features.yaml，并在私有特征目录写出 `_feature_registry.json`；
其中逐列登记版本、负责人、源字段、时间可用性和对应单测。

训练报告包含 PR-AUC、概率输入的 ROC-AUC、KS、Brier/ECE、告警预算、固定 FPR 召回、
月度/类型/新账户切片、特征漂移、推理吞吐和进程/GPU 内存。Bootstrap 默认关闭以避免
对全量测试集做隐式重采样；需要置信区间时显式指定至少 20 次迭代。

如需比较 hard-negative 训练，可先只对训练期生成 table OOF，再将该私有 OOF 文件传给
aml-train-table 的 --hard-negative-oof。采样会保留所有正例、优先选择 OOF 分数最高的
负例并用稳定哈希补足预算；验证和测试分数绝不能作为该输入。

## 后评分调查视图

冻结的交易分数可以进一步生成账户风险、严格时间递增的资金路径和关联子图案件视图：

    D:\Miniconda3\envs\aml-evidence\Scripts\aml-build-investigation-views.exe --transactions artifacts/pit_features --scores artifacts/fusion_test/test_fusion_scores.parquet --score-column fusion_calibrated_probability --as-of-ts 2023-08-23T23:59:59Z --output artifacts/test_investigation_views

该命令以 `transaction_id` 显式等值关联，要求每个分数都可在交易输入中找到；它会主动丢弃
`is_laundering` 和 `laundering_type`，因此不会把交易确认标签转写成账户或案件标签。输出的
`account_risk.parquet`、`funds_paths.json` 与 `case_views.json` 都是私有调查产物，案件状态
仅为 `investigation_candidate`，仍须人工复核。

## 受控 API 与 Mock Demo

默认启动的是无私有数据的 Mock Demo：

    D:\Miniconda3\envs\aml-evidence\Scripts\aml-api.exe

浏览器访问 http://127.0.0.1:8000/demo。Demo 只使用虚构案例，不调用内部 v1 路由或
外部 LLM。

私有推理需同时设置 AML_FEATURE_ROOT、AML_TABLE_MODEL_DIR、AML_MODEL_VERSION 与
AML_INTERNAL_API_TOKEN。可选设置 AML_GRAPHSAGE_MODEL_PATH 与 AML_FUSION_DIR 以恢复
冻结的 GraphSAGE、融合器、验证期校准器和阈值。没有所需融合组件时服务会拒绝启动，
不会静默降级。

内部接口包括批量日期分区评分、Evidence Package 查询、调查草稿和人工复核记录。复核
只写审计记录，绝不在线更新模型。

## Golden Set 与 ECNU

默认 Golden smoke test 不调用外部服务：

    D:\Miniconda3\envs\aml-evidence\Scripts\aml-evaluate-golden.exe --cases golden/mock_cases.json --typologies knowledge/typologies --output artifacts/golden_summary.json

若已获组织审批，可设置 AML_LLM_ENABLED=true，并将 API key 仅保留在 ECNU_API_KEY
环境变量中，再追加 --use-llm。外发内容不含交易、账户、告警 ID、时间戳、金额或原始
特征值；返回内容经过事实校验。Golden 输出与 run manifest 会记录 Prompt 版本、模型、
延迟和服务商 token 用量。

## 本地验收

    D:\Miniconda3\envs\aml-evidence\python.exe -m ruff check src tests
    D:\Miniconda3\envs\aml-evidence\python.exe -m pytest

Docker 配置只启动 Mock Demo：

    docker compose -f docker-compose.demo.yml up --build
