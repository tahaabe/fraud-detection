-- Clean layer: train + test on one timeline, identity joined, partitioned + clustered
CREATE SCHEMA IF NOT EXISTS fraud_clean OPTIONS (location = 'europe-west1');

CREATE OR REPLACE TABLE fraud_clean.transactions
PARTITION BY TIMESTAMP_TRUNC(transaction_ts, MONTH)
CLUSTER BY ProductCD, card1, addr1
OPTIONS (description = 'IEEE-CIS train+test, identity joined. Timestamp anchor 2017-12-01 is technical only.')
AS
WITH txn AS (
  SELECT 'train' AS split, *
  FROM fraud_raw.train_transaction
  UNION ALL
  -- test has no isFraud: insert NULL at the same position (2nd column) so UNION ALL lines up
  SELECT 'test' AS split, TransactionID, CAST(NULL AS INT64) AS isFraud, * EXCEPT (TransactionID)
  FROM fraud_raw.test_transaction
),
idn AS (
  SELECT *, 1 AS has_identity FROM fraud_raw.train_identity
  UNION ALL
  SELECT *, 1 AS has_identity FROM fraud_raw.test_identity
)
SELECT
  TIMESTAMP_ADD(TIMESTAMP '2017-12-01', INTERVAL t.TransactionDT SECOND) AS transaction_ts,
  DIV(t.TransactionDT, 86400) AS day,
  -- ID-like codes stored as floats in the CSVs -> integers (clustering needs non-float keys)
  t.* REPLACE (
    CAST(t.card1 AS INT64) AS card1,
    CAST(t.card2 AS INT64) AS card2,
    CAST(t.card3 AS INT64) AS card3,
    CAST(t.card5 AS INT64) AS card5,
    CAST(t.addr1 AS INT64) AS addr1,
    CAST(t.addr2 AS INT64) AS addr2
  ),
  i.* EXCEPT (TransactionID, has_identity),
  IFNULL(i.has_identity, 0) AS has_identity
FROM txn AS t
LEFT JOIN idn AS i USING (TransactionID);

