"""Real-time fraud scoring API: raw transaction -> risk score, decision, SHAP reasons."""
import json
import os
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.serving.features import FeatureBuilder

ART = Path(os.environ.get("ARTIFACT_DIR", "artifacts/xgb_v2/final"))
SCHEMA_DIR = Path(os.environ.get("SCHEMA_DIR", "sql/schemas"))
TOP_K = 5

# ---- Load artifacts once, at startup ----
fb = FeatureBuilder(ART)
booster = xgb.Booster()
booster.load_model(str(ART / "model.json"))
thresholds = json.loads((ART / "thresholds.json").read_text())
metrics = json.loads((ART / "metrics.json").read_text())
MODEL_VERSION = f"xgb_{json.loads((ART / 'feature_list.json').read_text())['version']}_{metrics['mode']}"

if metrics.get("xgboost_version") != xgb.__version__:  # fail fast on version mismatch
    raise RuntimeError(f"Model trained with XGBoost {metrics.get('xgboost_version')}, "
                       f"runtime has {xgb.__version__}")

# Raw IEEE-CIS schema (same JSON schemas used for the BigQuery load) = accepted input fields
KNOWN = {f["name"] for name in ("train_transaction", "train_identity")
         for f in json.loads((SCHEMA_DIR / f"{name}.json").read_text())}

app = FastAPI(title="Fraud Risk Scoring API", version="2.0")


class Transaction(BaseModel):
    """Raw transaction. Required fields are declared; all other IEEE-CIS fields are optional extras."""
    model_config = ConfigDict(extra="allow")
    TransactionID: int | None = None
    TransactionDT: int = Field(..., ge=0, description="Seconds since the dataset reference point")
    TransactionAmt: float = Field(..., gt=0)
    ProductCD: str


class Reason(BaseModel):
    feature: str
    value: str | float | None
    contribution: float  # SHAP value in log-odds; > 0 pushes toward fraud


class ScoreResponse(BaseModel):
    transaction_id: int | None
    risk_score: float  # ranking score, NOT a calibrated probability (class weighting)
    decision: str      # approve / review / high_risk
    top_reasons: list[Reason]
    unknown_fields: list[str]
    model_version: str
    latency_ms: float


class PredictRequest(BaseModel):
    """Vertex AI prediction format."""
    instances: list[Transaction] = Field(..., min_length=1, max_length=1000)


def _decide(p):
    if p >= thresholds["high_risk"]:
        return "high_risk"
    if p >= thresholds["review"]:
        return "review"
    return "approve"


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "xgboost_version": xgb.__version__,
            "n_features": len(fb.features),
            "thresholds": {"high_risk": thresholds["high_risk"], "review": thresholds["review"]}}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction, explain: bool = False):
    """Score one transaction. SHAP reasons are computed for review/high_risk decisions,
    or for any decision when explain=true."""
    t0 = time.perf_counter()
    raw = txn.model_dump()
    M = fb.build_matrix([raw])
    dm = fb.dmatrix(M)
    p = float(booster.predict(dm)[0])
    decision = _decide(p)

    reasons = []
    if explain or decision != "approve":
        contrib = booster.predict(dm, pred_contribs=True)[0][:-1]  # last column = bias
        top = np.argsort(-np.abs(contrib))[:TOP_K]
        reasons = [Reason(feature=fb.features[i], value=fb.display(M[0, i], i),
                          contribution=round(float(contrib[i]), 4)) for i in top]

    return ScoreResponse(
        transaction_id=txn.TransactionID, risk_score=round(p, 6), decision=decision,
        top_reasons=reasons, unknown_fields=sorted(set(raw) - KNOWN),
        model_version=MODEL_VERSION, latency_ms=round((time.perf_counter() - t0) * 1000, 2))


@app.post("/predict")
def predict(req: PredictRequest):
    """Batch scoring in Vertex AI format: {"instances": [...]} -> {"predictions": [...]}. No SHAP."""
    M = fb.build_matrix([t.model_dump() for t in req.instances])
    scores = booster.predict(fb.dmatrix(M))
    return {"predictions": [{"risk_score": round(float(s), 6), "decision": _decide(float(s))}
                            for s in scores],
            "model_version": MODEL_VERSION}