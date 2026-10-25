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