"""Serving-time feature computation. MUST match sql/03_txn_features.sql exactly
(verified by src/serving/check_parity.py)."""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

IDENTITY_FIELDS = [f"id_{i:02d}" for i in range(1, 39)] + ["DeviceType", "DeviceInfo"]


def has_identity(t):
    """1 if the transaction carries any identity information (= identity row existed in training)."""
    return int(any(t.get(f) is not None for f in IDENTITY_FIELDS))


def _clean(v):
    """Normalize missing values (None, NaN, pd.NA) to None."""
    try:
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return v


def amt_decimals(amt):
    # BigQuery ROUND rounds half away from zero; for positive amounts that is floor(x + 0.5)
    milli = math.floor(amt * 1000 + 0.5)
    if milli % 10:
        return 3
    if milli % 100:
        return 2
    if milli % 1000:
        return 1
    return 0


def card6_clean(v):
    if v is None:
        return None
    return v if v in ("debit", "credit") else "other"


def browser_family(id_31):
    if id_31 is None:
        return None
    s = re.sub(r"\s+", " ", re.sub(r"[0-9.]+", "", id_31)).strip()
    return s or None


def email_match(p, r):
    if p is None or r is None:
        return None
    return int(p == r)


def row_features(t):
    """The 6 row-level features, computed from one raw transaction (dict)."""
    dt = int(t["TransactionDT"])
    return {
        "amt_decimals": amt_decimals(float(t["TransactionAmt"])),
        "hour_rel": (dt // 3600) % 24,
        "weekday_rel": (dt // 86400) % 7,
        "card6_clean": card6_clean(t.get("card6")),
        "browser_family": browser_family(t.get("id_31")),
        "email_match": email_match(t.get("P_emaildomain"), t.get("R_emaildomain")),
        "has_identity": has_identity(t),
    }
    


class FeatureBuilder:
    """Raw transactions -> model input (same columns, order and encoding as training)."""

    def __init__(self, artifact_dir):
        d = Path(artifact_dir)
        spec = json.loads((d / "feature_list.json").read_text())
        self.features = spec["features"]
        cats = json.loads((d / "categories.json").read_text())
        self.cat_dtypes = {c: pd.CategoricalDtype(v) for c, v in cats.items()}
        self.cat_names = cats                                   # code -> label
        self.cat_index = {c: {v: i for i, v in enumerate(vals)} for c, vals in cats.items()}  # label -> code
        self.feature_types = ["c" if c in cats else "q" for c in self.features]

    # --- Reference path (pandas): mirrors training exactly; used by the parity test ---
    def build(self, txns):
        rows = []
        for t in txns:
            t = {k: _clean(v) for k, v in t.items()}
            rows.append({**t, **row_features(t)})
        df = pd.DataFrame(rows).reindex(columns=self.features)
        for c in self.features:
            if c in self.cat_dtypes:
                df[c] = df[c].astype("string").astype(self.cat_dtypes[c])
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        return df

    # --- Fast path (numpy): used by the API; categories encoded as training codes ---
    def build_matrix(self, txns):
        M = np.full((len(txns), len(self.features)), np.nan, dtype=np.float32)
        for r, t in enumerate(txns):
            t = {k: _clean(v) for k, v in t.items()}
            t.update(row_features(t))
            for j, c in enumerate(self.features):
                v = t.get(c)
                if v is None:
                    continue
                if c in self.cat_index:
                    code = self.cat_index[c].get(str(v))  # unseen category -> stays NaN
                    if code is not None:
                        M[r, j] = code
                else:
                    try:
                        M[r, j] = float(v)
                    except (TypeError, ValueError):     # non-numeric garbage -> NaN
                        pass
        return M

    def dmatrix(self, M):
        return xgb.DMatrix(M, feature_names=self.features, feature_types=self.feature_types,
                           enable_categorical=True)

    def display(self, value, j):
        """Human-readable value of feature j (category label, number, or None)."""
        if np.isnan(value):
            return None
        c = self.features[j]
        if c in self.cat_names:
            return self.cat_names[c][int(value)]
        return round(float(value), 4)