from pathlib import Path
import pandas as pd

raw = Path("data/raw")
for f in ["train_transaction.csv", "train_identity.csv"]:
    p = raw / f
    cols = pd.read_csv(p, nrows=0).columns
    n = sum(len(c) for c in pd.read_csv(p, usecols=[0], chunksize=200_000))
    print(f"{f}: {n:,} rows x {len(cols)} cols | {p.stat().st_size/1e6:.0f} MB")

y = pd.read_csv(raw / "train_transaction.csv", usecols=["isFraud"])["isFraud"]
print(f"Fraud rate: {y.mean():.4%} ({y.sum():,} fraud)")