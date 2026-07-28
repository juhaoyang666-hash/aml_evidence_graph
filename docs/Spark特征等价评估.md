# Spark 特征等价评估

> 本报告是 Windows 本地 Spark smoke 重放，不是生产集群 SLA。

- 结果：**通过**
- 全量重放：扫描/输出 355,998/355,998 行，17.06 秒，峰值进程树内存 1181.5 MB
- 增量重放：扫描/输出 56,694/28,155 行，11.07 秒，峰值 818.8 MB
- Spark：`local[2]`，shuffle partitions=8，physical Exchange=4
- 对齐：`transaction_id` 显式等值连接；没有普通笛卡尔积。

| PIT 特征 | Match rate | 最大绝对误差 |
|---|---:|---:|
| sender_outgoing_count_1d | 1.000000 | 0 |
| sender_outgoing_count_7d | 1.000000 | 0 |
| sender_outgoing_count_30d | 1.000000 | 0 |
| receiver_incoming_count_7d | 1.000000 | 0 |
| relationship_count_7d | 1.000000 | 0 |

Windows 缺少 Hadoop native 写入组件，本机采用 Spark 计算后 Arrow driver 写出；
窗口聚合与 shuffle 由 Spark 执行。Linux/集群环境使用原生 Spark Parquet writer。
当前记录物理 Exchange 节点数，不宣称生产集群 shuffle bytes。
该结果只覆盖 5 个代表性历史计数特征和 smoke 数据，不替代全量官方 PIT 流水线。
