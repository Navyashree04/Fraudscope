"""
POST /retrain -- OPTIONAL demo-only retraining trigger.

IMPORTANT: this is a synchronous, in-process background task for demo
purposes only. A real production system would put retraining behind an
async job queue (e.g. Celery, an Argo/Airflow pipeline, or a managed
training job), with versioned artifacts, approval gates, and rollback --
not a single unauthenticated HTTP endpoint that mutates the serving
model on disk.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from app.dependencies import load_model_state
from app.schemas import RetrainAcceptedResponse
from src.train import train

router = APIRouter()


def _retrain_and_reload() -> None:
    train()
    # Reload the singleton so the running process picks up the new
    # artifact without a restart -- fine for a demo, not how you'd want
    # a real hot-swap to work (no atomicity/versioning guarantees here).
    load_model_state()


@router.post("/retrain", response_model=RetrainAcceptedResponse, status_code=202)
def retrain(background_tasks: BackgroundTasks) -> RetrainAcceptedResponse:
    background_tasks.add_task(_retrain_and_reload)
    return RetrainAcceptedResponse(
        status="accepted",
        message=(
            "Retraining started as a background task on public/synthetic "
            "data. This is a demo stub, not a production job queue."
        ),
    )
