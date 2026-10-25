"""Vertex AI training job: XGBoost on BigQuery features (feature list v2), artifacts to GCS."""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import xgboost
from google.cloud import bigquery, storage
from xgboost import XGBClassifier

from src.models.train import PARAMS, evaluate, make_categories, prepare

SPEC = json.loads(Path("src/features/feature_list_v2.json").read_text())


def load(project, location):
    cols = SPEC["metadata"] + [SPEC["label"]] + SPEC["features"]
    sql = (f"SELECT {', '.join(cols)} FROM `{project}.fraud_features.{SPEC['source_table']}` "
           f"WHERE split = 'train' AND day BETWEEN 31 AND 182")
    client = bigquery.Client(project=project, location=location)
    df = client.query(sql).to_dataframe(create_bqstorage_client=True)
    cats = set(SPEC["categorical"])
    for c in SPEC["features"]:
        if c not in cats:
            df[c] = df[c].astype("float32")
    df[SPEC["label"]] = df[SPEC["label"]].astype("int8")
    print(f"Loaded {df.shape}", flush=True)
    return df


def upload_dir(local_dir, gcs_uri):
    bucket_name, prefix = gcs_uri.removeprefix("gs://").split("/", 1)
    bucket = storage.Client().bucket(bucket_name)
    for f in Path(local_dir).iterdir():
        bucket.blob(f"{prefix.rstrip('/')}/{f.name}").upload_from_filename(str(f))
        print(f"Uploaded {gcs_uri.rstrip('/')}/{f.name}", flush=True)


def clean(d):
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="europe-west1")
    ap.add_argument("--mode", choices=["validate", "final"], required=True)
    ap.add_argument("--output", required=True, help="gs://bucket/path (mode is appended)")
    ap.add_argument("--n-estimators", type=int, default=2000)
    a = ap.parse_args()

    df = load(a.project, a.location)
    feats, catcols = SPEC["features"], SPEC["categorical"]
    t0 = time.time()

    if a.mode == "validate":  # same protocol as the local experiments
        tr = df[df["day"] < 109]
        es = df[(df["day"] >= 109) & (df["day"] <= 122)]
        va = df[df["day"] >= 153]
        cats = make_categories(tr, catcols)
        Xtr, Xes, Xva = (prepare(d, feats, cats) for d in (tr, es, va))
        ytr, yes, yva = (d["isFraud"].to_numpy() for d in (tr, es, va))
        spw = (ytr == 0).sum() / (ytr == 1).sum()
        model = XGBClassifier(**PARAMS, scale_pos_weight=spw)
        model.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=200)
        p = model.predict_proba(Xva)[:, 1]
        metrics = {"best_iteration": int(model.best_iteration),
                   **evaluate(yva, p, va["ProductCD"].to_numpy())}
    else:  # final: all labeled days after warm-up, fixed number of trees
        cats = make_categories(df, catcols)
        X, y = prepare(df, feats, cats), df["isFraud"].to_numpy()
        spw = (y == 0).sum() / (y == 1).sum()
        params = {k: v for k, v in PARAMS.items() if k != "early_stopping_rounds"}
        params["n_estimators"] = a.n_estimators
        model = XGBClassifier(**params, scale_pos_weight=spw)
        model.fit(X, y, verbose=False)
        metrics = {"n_estimators": a.n_estimators, "train_rows": len(df), "train_days": [31, 182]}

    metrics = clean({"mode": a.mode, "feature_list": SPEC["version"], "n_features": len(feats),
                     "scale_pos_weight": spw, "xgboost_version": xgboost.__version__,
                     "train_minutes": (time.time() - t0) / 60, **metrics})
    print(json.dumps(metrics, indent=2), flush=True)

    out = Path(tempfile.mkdtemp())
    model.save_model(out / "model.json")
    (out / "categories.json").write_text(
        json.dumps({c: list(dt.categories) for c, dt in cats.items()}, indent=2))
    (out / "feature_list.json").write_text(json.dumps(SPEC, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    upload_dir(out, f"{a.output.rstrip('/')}/{a.mode}")


if __name__ == "__main__":
    main()