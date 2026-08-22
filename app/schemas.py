"""
Pydantic request/response schemas for FraudScope's API.

Feature fields (Time, V1-V28, Amount) mirror the public/synthetic dataset's
PCA-anonymized schema -- no raw cardholder or merchant fields are ever
part of this contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionRequest(BaseModel):
    """A single transaction to be scored.

    Matches the public Kaggle Credit Card Fraud Detection dataset's
    (or the synthetic-for-demo stand-in's) feature schema.
    """

    transaction_id: Optional[str] = Field(
        default=None,
        description="Optional passthrough ID for correlating with the "
                     "calling system's (e.g. TransactionWatch) alert record.",
    )
    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset window.")
    Amount: float = Field(..., ge=0, description="Transaction amount, non-negative.")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    @field_validator("Time")
    @classmethod
    def time_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Time must be non-negative")
        return v

    def to_ordered_dict(self) -> dict:
        """Return fields as a plain dict keyed exactly like FEATURE_COLUMNS."""
        d = self.model_dump(exclude={"transaction_id"})
        return d

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "txn_12345",
                "Time": 40000.0,
                "Amount": 149.62,
                **{f"V{i}": 0.0 for i in range(1, 29)},
            }
        }
    )


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: str


class ScoreResponse(BaseModel):
    transaction_id: Optional[str] = None
    risk_score: float = Field(..., ge=0, le=1)
    risk_band: str
    top_features: list[FeatureContribution]
    model_version: str
    scored_at: datetime
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_s: float


class RetrainAcceptedResponse(BaseModel):
    status: str
    message: str
