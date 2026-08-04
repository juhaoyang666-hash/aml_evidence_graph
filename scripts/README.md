# 脚本目录

脚本按用途分类，均建议从仓库根目录执行。稳定的产品入口仍以 `pyproject.toml` 中注册的 CLI 为准；本目录主要承载离线实验、数据审计、报告生成与运维辅助任务。

| 目录 | 用途 | 示例 |
| --- | --- | --- |
| `data/` | 数据审计、特征重放与 Spark 等价性验证 | `audit_saml_degree_proxy.py` |
| `experiments/` | 模型训练、消融、压力测试与实验汇总 | `run_gat_validation_candidate.py` |
| `retrieval/` | Typology 检索、盲审和拒答门控评估 | `evaluate_retrieval.py` |
| `reporting/` | MLflow、简历证据、LLM Holdout/脱敏聚合和服务基准报告 | `build_llm_holdout_golden.py`、`publish_llm_evidence.py` |
| `operations/` | 环境检查、API/Agent 基准和发布验证 | `verify_resume_release.py` |
| `pipelines/` | 串联多个步骤的 PowerShell/Shell/Python 流水线 | `run_full_train_chain.sh` |

新增文件应放入最贴近其主要职责的子目录。只有共享包标识和本说明保留在 `scripts/` 根目录；流水线引用其他脚本时应写完整分类路径。
