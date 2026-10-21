import json
from pathlib import Path
import pandas as pd
from src.data.load import RAW, TXN_CAT

OUT = Path("sql/schemas")
INT_COLS = {"TransactionID", "TransactionDT", "isFraud"}


def field(name, bq_type):
    return {"name": name, "type": bq_type, "mode": "NULLABLE"}


def txn_type(c):
    if c in INT_COLS:
        return "INT64"
    return "STRING" if c in TXN_CAT else "FLOAT64"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Transaction tables: types from our EDA rules
    for split in ("train", "test"):
        cols = pd.read_csv(RAW / f"{split}_transaction.csv", nrows=0).columns
        schema = [field(c, txn_type(c)) for c in cols]
        (OUT / f"{split}_transaction.json").write_text(json.dumps(schema, indent=2))

    # Identity tables: infer types from the full train identity file (small)
    idn = pd.read_csv(RAW / "train_identity.csv")
    types = {c: ("INT64" if c in INT_COLS
                 else "FLOAT64" if pd.api.types.is_numeric_dtype(idn[c])
                 else "STRING") for c in idn.columns}
    for split in ("train", "test"):
        cols = [c.replace("id-", "id_")
                for c in pd.read_csv(RAW / f"{split}_identity.csv", nrows=0).columns]
        schema = [field(c, types[c]) for c in cols]
        (OUT / f"{split}_identity.json").write_text(json.dumps(schema, indent=2))

    for f in sorted(OUT.glob("*.json")):
        n = len(json.loads(f.read_text()))
        print(f"{f.name}: {n} columns")


if __name__ == "__main__":
    main()