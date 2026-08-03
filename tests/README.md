# 测试目录

测试按系统职责分类，`pytest` 会递归发现全部 `test_*.py` 文件。

| 目录 | 覆盖范围 |
| --- | --- |
| `api/` | API、模型加载、多进程存储与并发租约 |
| `data/` | 数据契约、配置、时间切分、ODPS 与 Spark 重放 |
| `features/` | PIT 特征、冷启动特征、规则和特征构建 |
| `models/` | 表格/图模型、融合、OOF、采样和训练配置 |
| `investigation/` | 证据包、Controlled Agent、检索、Golden Set 与审计 |
| `evaluation/` | 指标、漂移、MLflow、候选对照和报告生成 |
| `engineering/` | CLI、设置和 Mock 端到端工程检查 |

新增测试应与被测模块放在同一职责分类中。跨模块端到端测试放入 `engineering/`，不要重新堆积到 `tests/` 根目录。
