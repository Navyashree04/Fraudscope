"""
Example client: simulates Project 2 ("TransactionWatch", Java/Spring)
calling FraudScope's /score endpoint over REST for a flagged/borderline
transaction.

This uses a made-up example transaction, not real transaction data.

Run FraudScope first (uvicorn app.main:app or docker compose up), then:
    python scripts/example_client.py
"""

from __future__ import annotations

import json
import sys

import os

import httpx
import pandas as pd

FRAUDSCOPE_URL = "http://localhost:8000/score"
DATA_PATH = os.environ.get("DATA_PATH", "data/creditcard.csv")


def build_sample_transaction() -> dict:
    """An example transaction, structured like the public/synthetic dataset
    schema. Not real transaction data.

    If the local dataset (public or synthetic-for-demo) is available, pulls
    one actual fraud-labeled row from it so the demo shows a realistic
    high-risk score; otherwise falls back to a plain illustrative example.
    """
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        fraud_rows = df[df["Class"] == 1]
        if len(fraud_rows) > 0:
            row = fraud_rows.iloc[0]
            payload = {"transaction_id": "txn_watch_00042"}
            payload.update({col: float(row[col]) for col in df.columns if col != "Class"})
            return payload

    payload = {
        "transaction_id": "txn_watch_00042",
        "Time": 86_412.0,
        "Amount": 872.50,
    }
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0
    return payload


def main() -> None:
    payload = build_sample_transaction()
    print("Simulated TransactionWatch -> FraudScope call")
    print(f"POST {FRAUDSCOPE_URL}")
    print(json.dumps(payload, indent=2))

    try:
        response = httpx.post(FRAUDSCOPE_URL, json=payload, timeout=10.0)
    except httpx.ConnectError:
        print(
            "\nCould not connect to FraudScope. Start it first, e.g.:\n"
            "  uvicorn app.main:app --reload\n"
            "or:\n"
            "  docker compose up\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nResponse status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
