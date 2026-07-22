-- AML Point-in-Time feature template for ODPS.
-- Safety invariant: ordinary Cartesian products are prohibited.
-- Replace __TRANSACTION_TABLE__ and __RUN_DATE__ with controlled private values.

WITH current_transactions AS (
  SELECT
    transaction_id,
    event_ts,
    sender_account_id,
    receiver_account_id,
    amount
  FROM __TRANSACTION_TABLE__
  WHERE ds = __RUN_DATE__
),
historical_sender_7d AS (
  SELECT
    sender_account_id,
    COUNT(1) AS sender_txn_count_7d,
    SUM(amount) AS sender_amount_sum_7d
  FROM __TRANSACTION_TABLE__
  WHERE event_ts >= DATE_SUB(__RUN_DATE__, 7)
    AND event_ts < __RUN_DATE__
  GROUP BY sender_account_id
)
SELECT
  current.transaction_id,
  current.event_ts,
  current.amount,
  COALESCE(history.sender_txn_count_7d, 0) AS sender_txn_count_7d,
  COALESCE(history.sender_amount_sum_7d, 0) AS sender_amount_sum_7d
FROM current_transactions AS current
LEFT JOIN historical_sender_7d AS history
  ON current.sender_account_id = history.sender_account_id;
