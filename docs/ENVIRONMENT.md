# AML 环境与运行约定

## 当前运行环境（Linux，全量实验所用）

全量 PIT、训练与评估在 Linux 服务器上执行，环境为 conda `risk`：

    PY=/data1/yangjuhao/envs/risk/bin/python   # Python 3.12
    cd /data1/yangjuhao/反洗钱/aml_evidence_graph
    export PYTHONPATH=src

- PyTorch **2.5.1+cu121**（官方 `download.pytorch.org/whl/cu121`），CUDA 可用
- GPU：8× NVIDIA RTX 3090；GraphSAGE 默认最多用 **4 张**
  （`--max-gpus`，或 `CUDA_VISIBLE_DEVICES` 限制）
- torch-geometric 2.8 + `pyg_lib` / `torch_scatter` / `torch_sparse`（pt25cu121）
- 长任务须在 **tmux** 会话中运行，便于断连后继续
- 链路脚本：`scripts/run_full_train_chain.sh`、`scripts/run_remaining_gpu.sh`

## 历史环境（Windows，早期开发）

早期在 Windows 下使用 conda `aml-evidence`（cu121 PyTorch 2.5.1，RTX 2060）。
本文档与 `FULL_RUN_AFTER_PIT.md` 中的 PowerShell 命令属于该环境，
在 Linux 上请改用上面的 `$PY -m ...` 形式。

主训练数据是公开合成数据集 **SAML-D**（`../data/SAML-D.csv`）。不要把它包装成
私有/真实业务数据。本地 API 令牌仍必须是至少 32 个字符的随机值；示例配置文件中的
空值不是可用占位符。不要在项目文件、命令历史、Notebook 或版本库中保存外部服务密钥。

新项目的处理顺序如下：

1. aml-profile-data 读取 SAML-D CSV，只生成聚合数据清单。
2. aml-convert-saml-d（别名：aml-convert-transactions；兼容旧名 aml-convert-private-data）
   写出按 event_date 和固定时间切分分区的 Parquet。
3. aml-build-pit-features 逐日构建因果特征。每笔交易只使用 [t-window, t) 的
   历史；同一时间戳的交易先全部评分，之后才共同进入历史。
4. aml-train-table 仅使用训练/验证期拟合，之后一次性读取完整测试期作评估，并报告
   固定召回下相对规则基线的告警削减率。
5. aml-train-graphsage 建立有向日度图快照。当前日的边只从过去日期的边采样邻居，
   训练期外的账户不进入已学习节点映射，而落入固定哈希桶；测试期标签绝不参与图消息。
   `configs/models.yaml` 的 `graphsage.architecture` 可选 graphsage/gat/rgcn/pna。
6. 融合时只能使用训练期 OOF 分数；概率校准与告警阈值只能在验证期选择。测试期只作
   一次完整评估，不允许重采样、调参或阈值搜索。
7. aml-build-investigation-views 在冻结评分之后，按明确 `as_of_ts` 聚合账户风险、资金路径
   和关联子图调查视图。它不接受标签列，也不将这些聚合结果反馈到交易模型。

如果配置了 AML_FEATURE_ROOT 与 AML_TABLE_MODEL_DIR，服务会切换到受控本地推理模式，
并要求 AML_INTERNAL_API_TOKEN；二者只配置其一会直接拒绝启动。Mock Demo 不需要
该令牌，且不读取本地完整产物。

受控评分默认使用表格 CatBoost 产物和 AML_ALERT_THRESHOLD。若同时配置
AML_GRAPHSAGE_MODEL_PATH 与 AML_FUSION_DIR，服务会恢复冻结的 GraphSAGE、OOF 融合器、
验证期校准器和验证期锁定阈值；融合器要求的任一组件缺失时会拒绝启动，而不会静默降级。

运行产物只写到 artifacts/，其内容已被 Git 排除。公开演示将只使用 Mock 数据和聚合结果，
绝不复制完整交易流水、模型产物或密钥。

## 可选 ECNU 调查注释

默认值 AML_LLM_ENABLED=false。启用后，系统仅调用 ECNU 的 ecnu-max 生成非事实性
调查注释；风险评分、规则阈值、模型融合和告警排序仍是本地确定性代码。API key 仅从
当前进程的 ECNU_API_KEY 读取，不能写入文件：

    $env:AML_LLM_ENABLED = "true"
    $env:AML_LLM_BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
    $env:AML_LLM_MODEL = "ecnu-max"

发送到外部服务的内容会移除 alert ID、transaction ID、时间戳、来源版本和原始特征
数值。返回内容必须只引用白名单 Evidence 字段，且不得带入数字、日期或实体标识；
校验失败时报告状态为 rejected_facts，网络失败时降级为本地确定性模板。

Golden Set 会记录每案端到端延迟、Prompt 版本、模型名和服务商响应中的 token 用量。
只有在组织已确认合同价格后，才设置两个 AML_LLM_*_COST_PER_MILLION_TOKENS_USD
变量以计算美元估算；未设置时报告保持为空，绝不凭空填写成本。
默认 Prompt 文件是 configs/prompts/ecnu-risk-evidence-v1.yaml；任何内容或参数变更都必须
新建版本文件，并重跑 Golden Set。其哈希会写入 Golden run manifest。
