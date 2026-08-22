"""
Feature schema and preprocessing shared by training and serving.

NOTE: V1-V28 are PCA components from the public/synthetic dataset -- they
are already dimensionality-reduced and anonymized upstream, so no raw
cardholder or merchant fields ever pass through this code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS


def validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if the dataframe doesn't match the expected schema."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")


def to_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Select and order columns exactly as the model expects.

    Time and Amount are used as-is (raw scale) here; the gradient-boosted
    tree model is scale-invariant so no explicit standardization is
    required, unlike a linear/distance-based baseline would need.
    """
    validate_columns(df)
    return df[FEATURE_COLUMNS].to_numpy(dtype=float)


def single_transaction_to_vector(payload: dict) -> np.ndarray:
    """Convert a single request payload (dict) into a model-ready feature vector."""
    row = {col: payload[col] for col in FEATURE_COLUMNS}
    return np.array([row[col] for col in FEATURE_COLUMNS], dtype=float).reshape(1, -1)
