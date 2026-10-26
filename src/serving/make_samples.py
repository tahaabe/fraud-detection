"""Save a few real raw transactions (validation month) as JSON request examples."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()
PROJECT = os.environ["GCP_PROJECT_ID"]
OUT = Path("tests/samples")

SQL = """
(SELECT * FROM fraud_clean.transactions
 WHERE split = 'train' AND day >= 153 AND isFraud = 1 AND ProductCD = 'C' ORDER BY TransactionID LIMIT 1)
UNION ALL
(SELECT * FROM fraud_clean.transactions
 WHERE split = 'train' AND day >= 153 AND isFraud = 0 AND ProductCD = 'W' ORDER BY TransactionID LIMIT 1)
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = bigquery.Client(project=PROJECT, location="europe-west1").query(SQL).to_dataframe()
    labels = df["isFraud"].tolist()
    raw = df.drop(columns=["split", "day", "transaction_ts", "isFraud","has_identity"])  # what a client would send
    for rec, label in zip(json.loads(raw.to_json(orient="records")), labels):
        name = "txn_fraud.json" if label == 1 else "txn_legit.json"
        (OUT / name).write_text(json.dumps(rec, indent=2))
        print(f"Wrote {OUT / name} (TransactionID {rec['TransactionID']})")


if __name__ == "__main__":
    main()