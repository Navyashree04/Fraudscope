"""
Synthetic-for-demo fallback dataset generator.

NOTE: This generates a SYNTHETIC stand-in for the public Kaggle
"Credit Card Fraud Detection" dataset. It is NOT real bank or customer
data, and it is not even the real Kaggle dataset -- it is a locally
generated approximation used only because this build environment cannot
reach Kaggle over the network. Every artifact produced from this data
(model, metrics, reports) must be labeled synthetic-for-demo.

Schema matches the real dataset exactly: Time, V1..V28, Amount, Class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_FEATURES = 28
FRAUD_RATE = 0.0017  # matches real dataset's ~0.17% positive rate


def generate_synthetic_creditcard_data(
    n_rows: int = 50_000,
    fraud_rate: float = FRAUD_RATE,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic-for-demo stand-in for creditcard.csv.

    SYNTHETIC DATA -- not real bank or customer transactions. Schema
    mirrors the public Kaggle Credit Card Fraud Detection dataset so the
    rest of the pipeline (train/evaluate/serve) runs unmodified against
    either this stand-in or the real Kaggle CSV.
    """
    rng = np.random.default_rng(random_state)

    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    n_legit = n_rows - n_fraud

    # PCA components (V1-V28): legitimate transactions ~ standard normal,
    # mirroring the real dataset's roughly-zero-mean PCA components.
    legit_v = rng.normal(loc=0.0, scale=1.0, size=(n_legit, N_FEATURES))
    fraud_v = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, N_FEATURES))

    # Shift a handful of components for fraud rows so there is real,
    # learnable signal (the real dataset has this kind of separability
    # in a subset of its PCA components, e.g. V4, V11, V14, V17).
    signal_features = [3, 10, 13, 16, 17]  # zero-indexed -> V4, V11, V14, V17, V18
    for idx in signal_features:
        fraud_v[:, idx] += rng.choice([-1, 1]) * rng.uniform(2.5, 4.5)

    # Amount: legit follows a right-skewed distribution typical of retail
    # spend; fraud skews toward smaller "test" charges with some large
    # outliers, a commonly cited real-world fraud pattern.
    legit_amount = rng.lognormal(mean=3.0, sigma=1.2, size=n_legit)
    fraud_amount = np.concatenate([
        rng.lognormal(mean=1.5, sigma=1.0, size=int(n_fraud * 0.8)),
        rng.lognormal(mean=5.5, sigma=0.5, size=n_fraud - int(n_fraud * 0.8)),
    ])
    rng.shuffle(fraud_amount)

    # Time: seconds elapsed across a synthetic ~2-day window.
    legit_time = np.sort(rng.uniform(0, 172_800, size=n_legit))
    fraud_time = np.sort(rng.uniform(0, 172_800, size=n_fraud))

    legit_df = pd.DataFrame(legit_v, columns=[f"V{i+1}" for i in range(N_FEATURES)])
    legit_df.insert(0, "Time", legit_time)
    legit_df["Amount"] = np.round(legit_amount, 2)
    legit_df["Class"] = 0

    fraud_df = pd.DataFrame(fraud_v, columns=[f"V{i+1}" for i in range(N_FEATURES)])
    fraud_df.insert(0, "Time", fraud_time)
    fraud_df["Amount"] = np.round(fraud_amount, 2)
    fraud_df["Class"] = 1

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df["Time"] = df["Time"].round(0)
    return df


if __name__ == "__main__":
    # Manual generation entry point: python -m src.synthetic_data
    out_path = "data/creditcard.csv"
    df = generate_synthetic_creditcard_data()
    df.to_csv(out_path, index=False)
    print(f"[synthetic-for-demo] wrote {len(df)} rows to {out_path}")
    print(df["Class"].value_counts())
