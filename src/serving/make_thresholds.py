"""Alert-budget thresholds: score quantiles on unlabeled recent traffic (test period)."""
import json
import os
from pathlib import Path

import numpy as np
import xgboost as xgb
from dotenv import load_dotenv
from google.cloud import bigquery

from src.models.train import prepare
from src.serving.features import FeatureBuilder

load_dotenv()
PROJECT = os.environ["GCP_PROJECT_ID"]
ART = Path("artifacts/xgb_v2/final")


def main():
    fb = FeatureBuilder(ART)
    sql = f"SELECT {', '.join(fb.features)} FROM fraud_features.model_table_v1 WHERE split = 'test'"
    client = bigquery.Client(project=PROJECT, location="europe-west1")
    df = client.query(sql).to_dataframe(create_bqstorage_client=True)
    for c in fb.features:
        if c not in fb.cat_dtypes:
            df[c] = df[c].astype("float32")

    booster = xgb.Booster()
    booster.load_model(str(ART / "model.json"))
    p = booster.predict(xgb.DMatrix(prepare(df, fb.features, fb.cat_dtypes), enable_categorical=True))

    th = {
        "high_risk": float(np.quantile(p, 0.99)),   # top 1% of traffic
        "review": float(np.quantile(p, 0.95)),      # top 5% of traffic
        "basis": "score quantiles on unlabeled test period (days 213-395)",
        "n_scored": int(len(p)),
    }
    (ART / "thresholds.json").write_text(json.dumps(th, indent=2))
    print(json.dumps(th, indent=2))


if __name__ == "__main__":
    main()