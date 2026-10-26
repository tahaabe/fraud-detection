"""Time each stage of scoring one transaction (median of 30 runs, after warm-up)."""
import json
import statistics
import time

import xgboost as xgb

from src.serving.features import FeatureBuilder

ART = "artifacts/xgb_v2/final"
fb = FeatureBuilder(ART)
booster = xgb.Booster()
booster.load_model(f"{ART}/model.json")
txn = json.loads(open("tests/samples/txn_fraud.json").read())


def timed(fn, n=30):
    fn()
    runs = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - s) * 1000)
    return f"{statistics.median(runs):7.1f} ms"


M = fb.build_matrix([txn])
dm = fb.dmatrix(M)
print("fast build features :", timed(lambda: fb.build_matrix([txn])))
print("fast make DMatrix   :", timed(lambda: fb.dmatrix(M)))
print("predict score       :", timed(lambda: booster.predict(dm)))
print("SHAP exact          :", timed(lambda: booster.predict(dm, pred_contribs=True)))
print("approve path total  :", timed(lambda: booster.predict(fb.dmatrix(fb.build_matrix([txn])))))