"""
POST /score -- scores a single transaction for fraud risk.

Reminder: this endpoint scores whatever transaction-shaped payload it is
given. It has no connection to any real bank, payment processor, or
customer database -- in this repo it is only ever exercised against
public/synthetic example data.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter

from app.dependencies import get_model_state
from app.schemas import FeatureContribution, ScoreResponse, TransactionRequest
from src.config import settings
from src.features import single_transaction_to_vector
from src.logging_utils import log_score_event

router = APIRouter()


def _risk_band(score: float) -> str:
    if score <= settings.risk_low_max:
        return "low"
    if score <= settings.risk_medium_max:
        return "medium"
    return "high"


@router.post("/score", response_model=ScoreResponse)
def score_transaction(payload: TransactionRequest) -> ScoreResponse:
    start = time.perf_counter()
    state = get_model_state()

    vector = single_transaction_to_vector(payload.to_ordered_dict())
    risk_score = float(state.model.predict_proba(vector)[0, 1])
    band = _risk_band(risk_score)

    top = state.explainer.top_features(vector, n=3)
    top_features = [
        FeatureContribution(feature=f["feature"], contribution=f["shap_value"], direction=f["direction"])
        for f in top
    ]

    latency_ms = (time.perf_counter() - start) * 1000.0

    log_score_event(
        transaction_id=payload.transaction_id,
        feature_summary={"Amount": payload.Amount, "Time": payload.Time},
        risk_score=risk_score,
        risk_band=band,
        model_version=state.model_version,
        latency_ms=latency_ms,
    )

    return ScoreResponse(
        transaction_id=payload.transaction_id,
        risk_score=round(risk_score, 6),
        risk_band=band,
        top_features=top_features,
        model_version=state.model_version,
        scored_at=datetime.now(timezone.utc),
        latency_ms=round(latency_ms, 3),
    )
