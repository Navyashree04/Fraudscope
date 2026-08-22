import joblib
import numpy as np

from src.config import FEATURE_COLUMNS, settings
from src.explain import Explainer


def test_top_features_shape_and_keys():
    model = joblib.load(settings.model_path)
    explainer = Explainer(model)

    vec = np.zeros((1, len(FEATURE_COLUMNS)))
    result = explainer.top_features(vec, n=3)

    assert len(result) == 3
    for item in result:
        assert set(item.keys()) == {"feature", "shap_value", "direction"}
        assert item["feature"] in FEATURE_COLUMNS
        assert item["direction"] in {"increases risk", "decreases risk"}


def test_top_features_direction_matches_sign_of_shap_value():
    model = joblib.load(settings.model_path)
    explainer = Explainer(model)

    vec = np.zeros((1, len(FEATURE_COLUMNS)))
    result = explainer.top_features(vec, n=5)

    for item in result:
        if item["shap_value"] > 0:
            assert item["direction"] == "increases risk"
        elif item["shap_value"] < 0:
            assert item["direction"] == "decreases risk"


def test_top_features_sorted_by_absolute_magnitude():
    model = joblib.load(settings.model_path)
    explainer = Explainer(model)

    vec = np.zeros((1, len(FEATURE_COLUMNS)))
    result = explainer.top_features(vec, n=5)

    magnitudes = [abs(item["shap_value"]) for item in result]
    assert magnitudes == sorted(magnitudes, reverse=True)
