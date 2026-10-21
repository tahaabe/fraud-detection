-- Behavioral features per pseudo user ID (uid). All windows use ONLY past transactions.
CREATE SCHEMA IF NOT EXISTS fraud_features OPTIONS (location = 'europe-west1');

CREATE OR REPLACE TABLE fraud_features.uid_features
PARTITION BY TIMESTAMP_TRUNC(transaction_ts, MONTH)
CLUSTER BY uid
AS
WITH base AS (
  -- Complete uids only: card1 + addr1 + card start day (day - D1)
  SELECT
    TransactionID, TransactionDT, TransactionAmt, P_emaildomain, DeviceInfo,
    CONCAT(CAST(card1 AS STRING), '_', CAST(addr1 AS STRING), '_',
           CAST(day - CAST(ROUND(D1) AS INT64) AS STRING)) AS uid
  FROM fraud_clean.transactions
  WHERE card1 IS NOT NULL AND addr1 IS NOT NULL AND D1 IS NOT NULL
),
firsts AS (
  -- Flag the first time a uid uses each email domain / device (to count distinct values so far)
  SELECT *,
    IF(P_emaildomain IS NOT NULL AND ROW_NUMBER() OVER (
         PARTITION BY uid, P_emaildomain ORDER BY TransactionDT, TransactionID) = 1, 1, 0) AS new_email,
    IF(DeviceInfo IS NOT NULL AND ROW_NUMBER() OVER (
         PARTITION BY uid, DeviceInfo ORDER BY TransactionDT, TransactionID) = 1, 1, 0) AS new_device
  FROM base
),
feats AS (
  SELECT
    TransactionID,
    uid,
    ROW_NUMBER() OVER w AS uid_txn_no,
    TransactionDT - LAG(TransactionDT) OVER w AS uid_secs_since_prev,
    COUNT(*) OVER (PARTITION BY uid ORDER BY TransactionDT
                   RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING) AS uid_cnt_24h,
    COUNT(*) OVER (PARTITION BY uid ORDER BY TransactionDT
                   RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING) AS uid_cnt_7d,
    AVG(TransactionAmt) OVER w_past AS uid_amt_mean_past,
    STDDEV_SAMP(TransactionAmt) OVER w_past AS uid_amt_std_past,
    SUM(new_email) OVER w_cum AS uid_n_emails,
    SUM(new_device) OVER w_cum AS uid_n_devices
  FROM firsts
  WINDOW
    w      AS (PARTITION BY uid ORDER BY TransactionDT, TransactionID),
    w_cum  AS (PARTITION BY uid ORDER BY TransactionDT, TransactionID
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    w_past AS (PARTITION BY uid ORDER BY TransactionDT, TransactionID
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
  t.TransactionID,
  t.transaction_ts,
  t.split,
  f.uid,
  f.uid_txn_no,
  f.uid_secs_since_prev,
  f.uid_cnt_24h,
  f.uid_cnt_7d,
  f.uid_amt_mean_past,
  f.uid_amt_std_past,
  SAFE_DIVIDE(t.TransactionAmt - f.uid_amt_mean_past, f.uid_amt_std_past) AS uid_amt_z,
  SAFE_DIVIDE(t.TransactionAmt, f.uid_amt_mean_past) AS uid_amt_ratio,
  f.uid_n_emails,
  f.uid_n_devices
FROM fraud_clean.transactions AS t
LEFT JOIN feats AS f USING (TransactionID);