# AML Evidence Graph

[![CI](https://github.com/juhaoyang666-hash/aml_evidence_graph/actions/workflows/ci.yml/badge.svg)](https://github.com/juhaoyang666-hash/aml_evidence_graph/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/juhaoyang666-hash/aml_evidence_graph?include_prereleases)](https://github.com/juhaoyang666-hash/aml_evidence_graph/releases)

基于公开合成数据集 **[SAML-D](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)**
（约 950 万笔交易）的交易级 AML 风险识别与调查辅助系统。

- **打分**：PIT 表格特征 + 历史图边分类 + 训练期 OOF 堆叠融合（验证期校准 / 锁定阈值）
- **调查**：Evidence Package 约束下的 Typology 检索、调查注释与 SAR 草稿；LLM **不参与**
  风险分数、阈值或案件结论
- **边界**：固定时间外切分；特征与图邻居只读预测时点之前的历史

完整指标、run_id 与复现说明见 **[docs/实验结果.md](docs/实验结果.md)** ·
**[docs/模型卡.md](docs/模型卡.md)**。

## 求职发布版：两分钟体验

无需 SAML-D 完整数据、GPU、密钥或外部 LLM。Docker 会使用清华 PyPI 镜像构建仅含
Mock/Schema/Typology 的公开演示：

```bash
git clone https://github.com/juhaoyang666-hash/aml_evidence_graph.git
cd aml_evidence_graph
docker compose -f docker-compose.demo.yml up --build --wait
```

打开 <http://127.0.0.1:8000/demo>，点击“加载 Mock 调查草稿”。演示重点：

1. 页面首先声明虚构数据和非业务结论边界；
2. Evidence Package 只提供已存在的规则、特征和图证据引用；
3. 确定性调查链生成事实摘要、待核实项和 SAR 骨架；
4. 状态停在 `draft_requires_human_review`，不会自动申报或处置。

验收与停止：

```bash
docker compose -f docker-compose.demo.yml ps
docker compose -f docker-compose.demo.yml down

# 已安装 Python 环境也可直接执行同一套发布断言
python scripts/verify_resume_release.py
```

发布烟雾会验证 API 版本、健康检查、Demo 边界、Mock Evidence、人工复核停点和
Mock 模型标识；它不读取 `artifacts/`，结果不能替代下文 SAML-D 时间外测试指标。

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

同协议边 GNN：GAT 0.948 > RGCN 0.903 > GraphSAGE 0.878 > PNA 0.705；
多关系 RGCN（R=4）消融 **0.887**（未超过单关系 RGCN / GAT）。

## 文档入口

完整索引见 **[docs/README.md](docs/README.md)**。常用入口：

| 文档 | 内容 |
|---|---|
| [实验结果.md](docs/实验结果.md) | 全量指标表、融合对比、架构对比、Bootstrap CI、Golden、调查视图 |
| [模型卡.md](docs/模型卡.md) | 模型边界、评价协议、限制 |
| [简历证据.md](docs/简历证据.md) | 产物自动门禁生成的唯一公开指标证据页 |
| [主架构图.md](docs/主架构图.md) | 风险评分、后评分聚合、受控 Agent 与人工审批边界 |
| [项目改进进度.md](docs/项目改进进度.md) | 求职改进完成度、sidecar 边界与下一步 |
| [特征工程V2实验.md](docs/特征工程V2实验.md) | FE v2 sidecar 完成结果与判定 |
| [面试要点.md](docs/面试要点.md) | 简历项目描述与面试自答要点 |
| [环境配置.md](docs/环境配置.md) | conda `risk`、GPU、全量复现与轻量验证 |

## 环境（Linux，全量实验）

```bash
export PY=/data1/yangjuhao/envs/risk/bin/python   # Python 3.12 · torch 2.5.1+cu121
cd /data1/yangjuhao/反洗钱/aml_evidence_graph
export PYTHONPATH=src
```

- GPU：最多 4× RTX 3090（`--max-gpus` / `CUDA_VISIBLE_DEVICES`）
- 长任务请放在 **tmux** 中
- 详细约定：[docs/环境配置.md](docs/环境配置.md)

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
# 后续 table / GAT / OOF / fusion 见 docs/环境配置.md 的全量复现清单
# 或沿用 configs/models.smoke.yaml + artifacts/models_smoke/
```

全量 OOF 使用默认 `--splits 3 --minimum-training-months 2`。  
**主线全量复现（GAT + catboost+GAT）**：[docs/环境配置.md](docs/环境配置.md)。

辅助脚本：`scripts/run_full_train_chain.sh`、`scripts/run_remaining_gpu.sh`、
`scripts/run_arch_comparison.sh`、`scripts/backfill_rule_hits.py`、
`scripts/tmux_launch_tier_ab.sh`（漂移/社区/无监督/序列/蒸馏等算力附录）。

## 调查视图 / API / Golden

冻结融合分后生成账户风险、资金路径与案件视图（主线目录
`artifacts/test_investigation_views_gat`），命令见 [环境配置.md](docs/环境配置.md)。

```bash
# Mock Demo（无完整交易）
aml-api   # 安装后的入口；浏览器 http://127.0.0.1:8000/demo

# Golden（30 案裁定集 + B5 扩容对抗探针；默认不调外部 LLM）
$PY -m aml_evidence_graph.investigation.golden \
  --cases golden/cases_v1.json --typologies knowledge/typologies \
  --output artifacts/golden_summary.json
# 可选：--use-llm（需 AML_LLM_ENABLED 与 ECNU_API_KEY，见 环境配置.md）

# 受控 Agent Golden（60 案，确定性无 LLM 基线）
$PY scripts/evaluate_agent_golden.py --overwrite
```

受控 Agent 的本地持久化可分别配置 checkpoint 与独立审计文件：

```bash
export AML_AGENT_CHECKPOINT_PATH=artifacts/checkpoints/investigations.sqlite
export AML_AGENT_AUDIT_PATH=artifacts/audit/investigation_events.sqlite
```

审计表只追加工具、节点和人工动作元数据，不保存证据正文、特征值、图边或复核备注正文；
SQLite 是本地原型，不应表述为生产 WORM 审计系统。

审批恢复支持 `Idempotency-Key`：相同键与相同请求可跨 checkpoint 重启回放，冲突正文返回
`409`。本机两 worker 已通过共享 SQLite 租约、续期和 fencing 验证；跨主机或高并发生产部署
仍需事务型共享协调与集中审计后端。

当前模板路径回归为 **34** 案（原 30 + 4 对抗扩容）；幻觉拦截 / 无证据拒答仍为 1.0，见
[大模型调查系统.md](docs/大模型调查系统.md)。
## 本地验收

```bash
export PYTHONPATH=src
$PY -m ruff check src tests
$PY -m pytest
python scripts/verify_resume_release.py
docker compose -f docker-compose.demo.yml up --build --wait
```
