"""Feature list v2: row-level features only (decided from ablations round 2)."""
import json
from pathlib import Path

v1 = json.loads(Path("src/features/feature_list_v1.json").read_text())
HISTORY = {"card1_cnt_30d", "addr1_cnt_30d", "p_email_cnt_30d", "amt_cnt_30d"}

features = [f for f in v1["features"] if not f.startswith("uid_") and f not in HISTORY]
v2 = {
    "version": "v2",
    "source_table": "model_table_v1",   # same BigQuery table, column subset
    "label": v1["label"],
    "metadata": v1["metadata"],
    "features": features,
    "categorical": [c for c in v1["categorical"] if c in features],
    "decision": "Row-level only: same PR-AUC as history features within seed noise, "
                "no history store needed at serving time (ablations round 2)",
}
Path("src/features/feature_list_v2.json").write_text(json.dumps(v2, indent=2))
print(f"v2: {len(features)} features ({len(v2['categorical'])} categorical)")