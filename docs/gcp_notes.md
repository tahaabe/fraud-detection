# GCP data layer

## Layers
- fraud_raw: 4 tables loaded from CSV (explicit schemas in sql/schemas/)
- fraud_clean.transactions: train + test (1,097,231 rows), identity joined, ID codes cast to INT64
  - Partitioned by month (TIMESTAMP_TRUNC(transaction_ts, MONTH)): 13 partitions
  - Clustered by ProductCD, card1, addr1
  - Timestamp anchor 2017-12-01 is technical only (supported by Dec 2017 / Dec 2018 volume peaks)

## Measured effects (dry runs)
- Columnar: 1 column over all rows = 8.8 MB read (table is several GB)
- Partition pruning: 1 month vs all months, same column → ~8% of rows read (~92% skipped)

## Constraints met
- CLUSTER BY requires non-float columns → cast card/addr codes to INT64
- PARTITION BY on a timestamp must use TIMESTAMP_TRUNC (not DATE_TRUNC(DATE(...)))
## Features: uid behavioral (sql/02_uid_features.sql → fraud_features.uid_features)
- uid = card1 + addr1 + (day − D1); complete keys only (incomplete → NULL features)
- Features: uid_txn_no, uid_secs_since_prev, uid_cnt_24h, uid_cnt_7d, uid_amt_mean_past,
  uid_amt_std_past, uid_amt_z, uid_amt_ratio, uid_n_emails, uid_n_devices
- Point-in-time correctness: window frames end at "1 PRECEDING" (or current row for counts
  that include the current transaction's own known values); no labels used
- Checks: row counts intact; no-uid rate 11.3% (train) = EDA; manual audit of one uid passed;
  fraud rate rises monotonically with uid_txn_no (2.0% → 4.8%, train, complete uids)
- z-score is NULL when < 2 past amounts or zero spread (identical past amounts) → left as NULL for XGBoost
## Features: transaction + entity (sql/03_txn_features.sql → fraud_features.txn_features)
- Row-level: amt_log, amt_decimals, hour_rel, weekday_rel (relative cycle position),
  card6_clean, browser_family (id_31 without version), email_match
- Rolling 30-day past-only counts: card1, addr1, P_emaildomain, exact amount (amt_milli key)
- Checks: amt_decimals shares and fraud rates identical to EDA (SQL = pandas);
  browser_family has no version numbers
- Cold start: history-based features are deflated in days 1–30 and in the first test month (30-day gap)
  → warm-up: days 1–30 used as history only, excluded as training rows
  → artifact of the dataset's edges; production history is continuous
- Possible improvement: normalize counts by total traffic (holiday volume inflates counts)
## Model table (src/gcp/build_model_table.py → sql/04_model_table.sql → fraud_features.model_table_v1)
- SQL generated from src/features/v_keep_v1.json; feature list written to src/features/feature_list_v1.json
  (single source of truth for training and API)
- 240 features (29 categorical) = base 220 + uid 10 + txn 10; metadata: TransactionID, transaction_ts, day, split, fold, uid
- Excluded from features: TransactionDT/day/timestamp (calendar time), uid string (ID), amt_log (redundant for trees)
- Folds: warmup 1–30 (history only), train 31–122, gap 123–152, valid 153–182, test 213–395
- Checks: 1,097,231 rows, no duplicate IDs, 247 columns, 20,663 frauds across labeled folds
- Train fold: 3.98% fraud → scale_pos_weight ≈ 24; valid: 3.50% (drift → tune threshold on valid)
- Limitation: warmup = December → holiday behavior absent from training rows (ablation: include warmup)
## IAM (least privilege)
- Compute default SA: roles/cloudbuild.builds.builder (project) → Cloud Build can read sources, push images, write logs
- Vertex AI service agent: roles/storage.objectViewer on the model bucket only → Model Registry can read artifacts