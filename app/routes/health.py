from __future__ import annotations

import time

from fastapi import APIRouter

from app.dependencies import get_model_state
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = get_model_state()
    return HealthResponse(
        status="ok",
        model_version=state.model_version,
        uptime_s=round(time.time() - state.started_at, 3),
    )
