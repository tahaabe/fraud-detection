# Model results

## Baseline v1 (240 features, XGBoost, lr 0.05, 1500 trees: NOT converged, best_iteration 1498)
- Validation: days 153–182 (30-day gap after training), past-only features
- PR-AUC 0.489 (base rate 0.035), ROC-AUC 0.883
- Top 1%: recall 24.6%, precision 86.2% | Top 5%: recall 53.4%, precision 37.5%
- non-W PR-AUC 0.707 (base 9.3%) vs W 0.127 (base 1.9%) → W is the weak regime (no identity data)
- Note: not comparable to Kaggle leaderboard (train+test aggregates, different period, no gap)