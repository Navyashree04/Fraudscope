"""
SHAP-based explainability wrapper for FraudScope.

Rebuilds a shap.TreeExplainer from the saved model at startup (simpler and
more robust than pickling the explainer itself, and cheap for tree models).
"""

from __future__ import annotations

import numpy as np
import shap

from src.config import FEATURE_COLUMNS


class Explainer:
    """Wraps a shap.TreeExplainer for a trained tree-based model."""

    def __init__(self, model):
        self.model = model
        self._explainer = shap.TreeExplainer(model)

    def top_features(self, feature_vector: np.ndarray, n: int = 3) -> list[dict]:
        """Return the top-n features contributing to the prediction for one row.

        feature_vector: shape (1, n_features), in the same column order as
        FEATURE_COLUMNS.

        Returns a list of dicts: {feature, shap_value, direction}, sorted by
        absolute SHAP value descending. direction is "increases risk" if the
        SHAP value is positive (pushes prediction toward fraud) or
        "decreases risk" if negative.
        """
        if feature_vector.ndim != 2 or feature_vector.shape[0] != 1:
            raise ValueError("feature_vector must have shape (1, n_features)")

        shap_values = self._explainer.shap_values(feature_vector)
        # For binary XGBClassifier, TreeExplainer returns shap values for the
        # positive class directly as a (1, n_features) array in modern shap.
        row = np.asarray(shap_values).reshape(-1)

        order = np.argsort(-np.abs(row))[:n]
        results = []
        for idx in order:
            value = float(row[idx])
            results.append({
                "feature": FEATURE_COLUMNS[idx],
                "shap_value": round(value, 6),
                "direction": "increases risk" if value > 0 else "decreases risk",
            })
        return results
