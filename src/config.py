"""
Central configuration for FraudScope.

Reminder: DATA_PATH and MODEL_PATH must always point at artifacts derived
from a PUBLIC or SYNTHETIC dataset -- never real bank or customer data.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Paths
    data_path: str = "data/creditcard.csv"
    model_path: str = "models/fraud_model_v1.joblib"
    model_metadata_path: str = "models/metadata.json"
    evaluation_report_path: str = "reports/evaluation_report.md"

    # Model
    model_version: str = "v1"
    random_state: int = 42
    test_size: float = 0.2

    # Risk bands (probability of fraud, 0-1). Anything <= risk_low_max is
    # "low", between risk_low_max and risk_medium_max is "medium", above
    # risk_medium_max is "high". These are demo defaults, not calibrated
    # against any real fraud-ops cost model.
    risk_low_max: float = 0.3
    risk_medium_max: float = 0.7

    # Logging
    log_level: str = "INFO"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent


settings = Settings()

# Feature schema shared by training, serving, and tests.
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", "Amount"]
FEATURE_COLUMNS = ["Time"] + PCA_FEATURES + ["Amount"]
TARGET_COLUMN = "Class"
