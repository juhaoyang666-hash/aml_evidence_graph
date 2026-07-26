# 批式特征任务说明（大数据思维加分）

头部 JD 常提 Hive/Spark。本仓库全量特征在单机 Python PIT 管线中完成
（~9.5M 行 / 321 日分区），**没有**集群 Spark 作业。下面用同一语义说明「批式特征」
应如何组织，便于面试迁移。

## 本项目实际做法

| 步骤 | 实现 |
|---|---|
| 输入 | 按 `event_date` 分区的 Parquet（`artifacts/prepared_transactions`） |
| 变换 | `aml-build-pit-features`：账户滚动窗口 + 关系窗口 + 规则命中 |
| 输出 | `artifacts/pit_features/event_date=*` + `_feature_registry.json` |
| 约束 | 仅 `[t−window, t)`；同秒隔离 |

等价心智模型（若放到数仓）：

```text
fact_transactions (partitioned by event_date)
  → daily account aggregates (map)
  → rolling merge by account_id / pair_id (stateful reduce, time-ordered)
  → join back to transaction grain as PIT features
```

## 可选本地批式验算（DuckDB）

在已有 PIT 上做**只读**聚合检查（不替换官方 PIT）：

```bash
export PY=/data1/yangjuhao/envs/risk/bin/python
$PY - <<'PY'
import duckdb
con = duckdb.connect()
# 示例：统计测试期附近分区行数（按机器上实际分区调整）
print(con.execute("""
  SELECT event_date, COUNT(*) AS n
  FROM read_parquet('artifacts/pit_features/event_date=*/*.parquet', hive_partitioning=true)
  GROUP BY 1 ORDER BY 1 DESC LIMIT 5
""").fetchdf())
PY
```

若环境无 DuckDB：`pip install duckdb`（勿写入密钥；勿提交 artifacts）。

## 面试怎么说

- **有**：千万级分区 Parquet、时间正确的滚动特征、特征登记与复现清单。  
- **没有**：公司级 Hive 仓、Spark 资源队列、实时特征存储。  
- **迁移**：协议（PIT、分区、只读历史）可映射到 Spark window / 增量状态；规模上去时把
  Python 滚动状态换成分布式引擎，**语义不变**。

## 已跑通的本地验算

见 [BATCH_FEATURE_REPLAY.md](BATCH_FEATURE_REPLAY.md)：DuckDB / Polars 重放 `sender_outgoing_count_7d`，对抽样子集与官方 PIT **match rate = 1.0**。
