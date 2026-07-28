# 批式 PIT 特征重放与大数据迁移说明

本项目的全量特征由单机 Python PIT 管线生成（约 950 万行、321 个日期分区），没有公司级
Hive 仓、Spark 资源队列或实时特征存储。本实验使用 DuckDB 和 Polars 只读重放一个代表性
窗口特征，验证向量化实现与官方 PIT 的语义一致性。

- 产物：`artifacts/batch_feature_replay/`
- 命令：`PYTHONPATH=src python scripts/run_batch_feature_replay.py`

## 当前批式数据组织

| 步骤 | 实现 |
|---|---|
| 输入 | `artifacts/prepared_transactions/event_date=*` 分区 Parquet |
| 变换 | 账户滚动窗口、关系窗口、图统计和规则命中 |
| 输出 | `artifacts/pit_features/event_date=*`、特征登记和 run manifest |
| 时间约束 | 每笔交易只读 `[t-window, t)`；同一时间戳交易相互不可见 |

等价的数仓处理结构：

```text
fact_transactions (partitioned by event_date)
  → daily account/pair aggregates
  → time-ordered rolling state by account_id/pair_id
  → explicit equality join back to transaction grain
  → partitioned PIT feature table
```

ODPS、Hive 或 Spark 实现应使用预聚合和显式等值连接，禁止普通笛卡尔积。

## 重放协议与结果

对 5 个 scored day 加 7 天历史窗口重放 `sender_outgoing_count_7d`，分别使用 DuckDB
`RANGE` window 和 Polars rolling，与官方 PIT 逐行对齐。

| 引擎 | Match rate vs PIT | 约 rows/s |
|---|---:|---:|
| DuckDB `RANGE` window | **1.0** | ~3.7e5 |
| Polars rolling | **1.0** | ~9.2e4 |

## 已验证的重建优化

初版单进程全量 PIT 构建约需 8 小时。规则阈值只依赖已有 PIT 列，因此规则版本更新不需要
重放历史窗口；`scripts/backfill_rule_hits.py` 可以直接补写命中列和规则证据。

| 方式 | 实测耗时 |
|---|---:|
| 全量 PIT 重建 | 约 8 小时 |
| 规则回填 dry run | 约 47 秒 |
| 规则回填并写入 | 约 2 分钟 |

脚本在规则引用特征缺失时拒绝执行，避免把真正需要历史重放的特征变更误当作简单回填。
后续若继续扩大数据规模，优先考虑增量窗口累加器、按账户哈希分桶并行和向量化窗口；任何
替换实现都必须重新验证 PIT 与同秒隔离语义。

## 如何用于面试

- 已实现：千万级分区 Parquet、严格时间窗口、特征登记、运行清单和两种向量化重放。
- 未实现：公司级 Hive/Spark 集群作业、资源队列、在线特征存储或生产 SLA。
- 可迁移部分：PIT 时间语义、日期分区、账户/关系 key、增量状态和回写交易粒度。
- 仍需重验：同秒隔离、迟到数据、分区回补、shuffle 成本和故障重试。

## 边界

该实验只覆盖一个特征和有限日期，不替换官方 Python PIT，也不能表述为已完成 Spark/Hive
生产迁移。它证明两种向量化引擎可以在该样本上复现同一时间协议。
