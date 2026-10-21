# EDA Decisions Log

Each line: finding → decision.

## Framing
- Row = one transaction; label from chargebacks propagated to the linked account
- Real-time scoring → features must use past transactions only (no train+test aggregates)

## Target
- Fraud rate 3.5% (20,663 / 590,540) → metric: PR-AUC; XGBoost scale_pos_weight ≈ 27

## Structure
- Identity present for 24.4% of rows (1-to-0/1 on TransactionID) → left join + has_identity flag
- Test set is later in time → time-based validation, no random split

## Missingness
- C, card, core columns complete → backbone features
- D1 ~0.2% missing → usable as base for pseudo user ID
- addr1/addr2 missing together (11.1%) → treat as a unit
- id columns 75.6–99.2% missing → drop near-empty id columns (threshold TBD in 2d)
- No imputation by default: XGBoost handles NaN natively; missingness may be a signal
- V: 339 cols in 15 NaN-count blocks → reduce per block in 2d (verify row-level patterns first)

## Time
- ~182 days; holiday volume peak around days 18–25 (if day 0 = Dec 1, community assumption)
- Weekly volume cycle → test weekday/hour features
- Weekly fraud rate drifts 1.8%–5.0% → time-based split mandatory; recalibrate threshold; monitor in prod
- amt_decimals: 2 decimals = 1.0% fraud, 3 decimals = 11.7% (likely overlaps product C) → keep as feature
- M1=F only 25 rows → ignore its rate; M1 ≈ "check performed" flag
- M4=M2 (59.9K rows, 11.4%) and M5 F<T (132K / 108K rows) → real signals
- Product W never has identity nor recipient email → separate data-collection regime
- has_identity effect is mostly a product proxy; residual effect within C (12.3% vs 5.8%)
  → single model, but evaluate W vs non-W separately
  ## Redundancy (2d)
- V columns: 15 NaN blocks confirmed at row level; blocks interleave in numbering (V138/V143, V167/V169, V217/V220)
- V reduction: greedy |Pearson| > 0.75 within block, keep max nunique → 339 → 138
  (saved: src/features/v_keep_v1.json) → ablation planned: all V vs reduced V
- Pearson only catches linear redundancy (limitation)
- 9 id columns ~99.1% missing (id_07, id_08, id_21–id_27): presence 8.2% vs 7.8% fraud within identity rows (not significant) → drop all, no flag
## Pseudo user ID (2d)
- D1 ≈ days since card first use: max D1 grows ~1/day over time; diagonal streaks per card → hypothesis supported
- uid = card1 + addr1 + (day − D1): 217,850 uids, median 1 txn, 57.5% single-txn uids (~21% of transactions without history)
- Purity of fraud groups: card1 3.6% → card1+addr1 8% → uid 39% (chance ≈ 2%) → uid approximates an account
- Risk rises with uid history length (2.8% first txn → 5.8% for 51+): partly label propagation, partly velocity behavior
- Behavioral features: past-only, activity-based (counts, amounts, timing) — never past labels (chargeback delay)
- Possible refinement: add P_emaildomain to the uid key (test later)
- Largest uids are complete keys (top: 1,414 txns) → heavy users or key collisions; P_emaildomain refinement may split collisions
- Negative D1n = card first used before the data window (normal)
- Incomplete uid (addr1 or D1 missing): 11.3% of rows, fraud 11.6% vs 2.5%
  → uid history features = NaN for incomplete uids; missingness signal kept via addr1 NaN
  ## Drift (2e)
- Train days 1–182, test days 213–395 (30-day gap) → validation: train early, gap, validate last month
- Test mix: more product C (11.6→13.7%), more identity (24.4→28%), more mobile (9.4→12.3%)
  → expect higher fraud rate in test; recalibrate threshold; handle unseen categories
- Adversarial AUC 0.907 (strong drift); top drifters: V9, id_31, D15/D11/D10/D6, C12, dist1
- Principle: remove artificial (calendar-encoding) drift, not natural drift; drift ≠ automatic drop
- Browser family (id_31 without version) → AUC 0.907 → 0.891 → adopt
- Removing V9 → no change (drift redistributes to correlated V columns) → V9 decided on predictive value, not drift
- Removing raw D → 0.848: D carries part of the drift, likely high predictive value → keep, test via ablation
- Remaining drift spread across many V blocks + id_13 → pervasive, real behavior change
  → strategy: time-based validation with gap; select features on future-period performance;
    monitor feature/score distributions and retrain periodically in production
- Hypothesis: uid-relative features (amount vs uid history) drift less than raw values