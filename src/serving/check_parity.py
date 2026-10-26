"""Parity test: serving features (from raw transactions) == training features (from BigQuery).
Checks the pandas reference path column by column, and the fast numpy path on predictions."""
import os
import sys
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from dotenv import load_dotenv
from google.cloud import bigquery

from src.models.train import prepare
from src.serving.features import FeatureBuilder

load_dotenv()
PROJECT = os.environ["GCP_PROJECT_ID"]
ART = "artifacts/xgb_v2/final"
SAMPLE = "split = 'train' AND MOD(TransactionID, 500) = 0"


def query(sql):
    client = bigquery.Client(project=PROJECT, location="europe-west1")
    return client.query(sql).to_dataframe(create_bqstorage_client=True)


def main():
    fb = FeatureBuilder(ART)
    feats, cats = fb.features, fb.cat_dtypes

    # Training path
    model_df = query(f"SELECT TransactionID, {', '.join(feats)} "
                     f"FROM fraud_features.model_table_v1 WHERE {SAMPLE}").sort_values("TransactionID")
    for c in feats:
        if c not in cats:
            model_df[c] = model_df[c].astype("float32")
    X_train = prepare(model_df, feats, cats).reset_index(drop=True)

    # Serving paths, from RAW transactions
    raw = query(f"SELECT * FROM fraud_clean.transactions WHERE {SAMPLE}").sort_values("TransactionID")
    # Only fields a real client could send: the raw IEEE-CIS schema (no label, no derived columns)
    schema = {f["name"] for n in ("train_transaction", "train_identity")
              for f in json.loads(Path(f"sql/schemas/{n}.json").read_text())} - {"isFraud"}
    records = [{k: v for k, v in r.items() if k in schema} for r in raw.to_dict("records")]
    X_serve = fb.build(records)
    M_fast = fb.build_matrix(records)

    bad = []
    for c in feats:
        if c in cats:
            ok = np.array_equal(X_train[c].cat.codes.to_numpy(), X_serve[c].cat.codes.to_numpy())
        else:
            ok = np.allclose(X_train[c].to_numpy(), X_serve[c].to_numpy(), equal_nan=True)
        if not ok:
            bad.append(c)

    booster = xgb.Booster()
    booster.load_model(f"{ART}/model.json")
    p_train = booster.predict(xgb.DMatrix(X_train, enable_categorical=True))
    p_serve = booster.predict(xgb.DMatrix(X_serve, enable_categorical=True))
    p_fast = booster.predict(fb.dmatrix(M_fast))
    diff_serve = float(np.abs(p_train - p_serve).max())
    diff_fast = float(np.abs(p_train - p_fast).max())

    print(f"Rows compared: {len(X_train)} | features: {len(feats)}")
    print(f"Mismatched features (pandas path): {bad if bad else 'none'}")
    print(f"Max prediction difference, pandas path: {diff_serve:.2e}")
    print(f"Max prediction difference, numpy path:  {diff_fast:.2e}")
    sys.exit(1 if bad or max(diff_serve, diff_fast) > 1e-6 else 0)


if __name__ == "__main__":
    main()