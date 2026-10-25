# Model results

## Baseline v1 (240 features, XGBoost, lr 0.05, 1500 trees: NOT converged, best_iteration 1498)
- Validation: days 153–182 (30-day gap after training), past-only features
- PR-AUC 0.489 (base rate 0.035), ROC-AUC 0.883
- Top 1%: recall 24.6%, precision 86.2% | Top 5%: recall 53.4%, precision 37.5%
- non-W PR-AUC 0.707 (base 9.3%) vs W 0.127 (base 1.9%) → W is the weak regime (no identity data)
- Note: not comparable to Kaggle leaderboard (train+test aggregates, different period, no gap)
## Ablations, round 2
- Seed noise is up to ~0.012 PR-AUC (base_only: 0.528 vs 0.516) → single-run differences < ~0.015 inconclusive
- All variants with/without engineered features fall in 0.51–0.53 → no measurable effect
- no_cumulative (0.511) ≈ baseline → calendar-drift hypothesis not supported
- Most consistent explanation: redundancy — Vesta's V/C/D were built with real IDs and full history
- Decision: row-level feature set (BASE + 6 row features): same accuracy, no history store needed at
  serving time, interpretable signals for SHAP. uid features kept in BigQuery as documented work.
- Lesson: estimate noise with several seeds before trusting any single-run difference
## Final feature set: v2 (row-level, 226 features)
- final_row: PR-AUC 0.518 / 0.521 (2 seeds), W 0.188 / 0.194, recall@5% 56.9% / 57.2%
- Stable across seeds (spread 0.003) vs base_only (0.012)
- Chosen over history-based set: equal accuracy within noise, no history store at serving time,
  interpretable row-level signals (card6, browser family, amount decimals) for SHAP
## Training infrastructure
- Training code containerized (docker/trainer.dockerfile, pinned versions) and pushed to Artifact Registry
- Vertex AI custom training blocked: free-trial quota = 1 CPU for N1/E2 in europe-west1, not adjustable
  (Vertex training machines need ≥ 2 vCPUs) → same script run locally, reading BigQuery, writing artifacts to GCS
- Final model: days 31–182, 2,000 trees (early stopping found ~1,700–1,850 on the dev split; ~2× data → slightly more)
- Model registration planned with the FastAPI serving image (Step 6)
## Official results (feature list v2, src/cloud/train_job.py)
- Validation (train 31–108, early stop 109–122, valid 153–182): PR-AUC 0.518, ROC-AUC 0.888,
  top 1%: precision 85.8% / recall 24.5%, top 5%: precision 40.1% / recall 57.2%,
  non-W 0.715 / W 0.192, best_iteration 1,465
- Final model: days 31–182 (456,201 rows), 2,000 trees, scale_pos_weight 25.4, XGBoost 3.2.0
- Artifacts: gs://…/models/xgb_v2/{validate,final}/ (model, categories, feature list, metrics)