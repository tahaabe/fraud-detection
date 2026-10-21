-- Transaction-level transforms + rolling 30-day entity frequencies (past transactions only)
CREATE OR REPLACE TABLE fraud_features.txn_features
PARTITION BY TIMESTAMP_TRUNC(transaction_ts, MONTH)
CLUSTER BY TransactionID
AS
WITH base AS (
  SELECT
    TransactionID, transaction_ts, split, day, TransactionDT, TransactionAmt,
    card1, addr1, card6, P_emaildomain, R_emaildomain, id_31,
    CAST(ROUND(TransactionAmt * 1000) AS INT64) AS amt_milli   -- exact amount as an integer key
  FROM fraud_clean.transactions
)
SELECT
  TransactionID,
  transaction_ts,
  split,

  -- Amount
  SAFE.LN(TransactionAmt) AS amt_log,
  CASE
    WHEN MOD(amt_milli, 10)   != 0 THEN 3
    WHEN MOD(amt_milli, 100)  != 0 THEN 2
    WHEN MOD(amt_milli, 1000) != 0 THEN 1
    ELSE 0
  END AS amt_decimals,

  -- Relative time (true start/timezone unknown: cycle position, not clock time)
  MOD(DIV(TransactionDT, 3600), 24) AS hour_rel,
  MOD(day, 7) AS weekday_rel,

  -- Categorical cleanups
  CASE WHEN card6 IN ('debit', 'credit') THEN card6
       WHEN card6 IS NULL THEN NULL
       ELSE 'other' END AS card6_clean,
  NULLIF(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(id_31, r'[0-9.]+', ''), r'\s+', ' ')), '') AS browser_family,
  CASE WHEN P_emaildomain IS NULL OR R_emaildomain IS NULL THEN NULL
       WHEN P_emaildomain = R_emaildomain THEN 1 ELSE 0 END AS email_match,

  -- Rolling 30-day frequencies (2,592,000 s = 30 days), past only; NULL key -> NULL
  IF(card1 IS NULL, NULL, COUNT(*) OVER (PARTITION BY card1 ORDER BY TransactionDT
       RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING)) AS card1_cnt_30d,
  IF(addr1 IS NULL, NULL, COUNT(*) OVER (PARTITION BY addr1 ORDER BY TransactionDT
       RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING)) AS addr1_cnt_30d,
  IF(P_emaildomain IS NULL, NULL, COUNT(*) OVER (PARTITION BY P_emaildomain ORDER BY TransactionDT
       RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING)) AS p_email_cnt_30d,
  COUNT(*) OVER (PARTITION BY amt_milli ORDER BY TransactionDT
       RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING) AS amt_cnt_30d
FROM base;