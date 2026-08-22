"""
FraudScope FastAPI service.

Loads the trained model artifact ONCE at startup (lifespan hook) and
serves /score and /health. Reminder: the served model is trained only on
a public or synthetic-for-demo dataset -- this service has no connection
to any real bank, cardholder, or transaction data.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.dependencies import load_model_state
from app.routes import health, retrain, score
from src.config import settings

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraudscope.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifact from %s ...", settings.model_path)
    state = load_model_state()
    logger.info(
        "Model loaded: version=%s dataset_is_synthetic=%s",
        state.model_version,
        state.metadata.get("dataset_is_synthetic"),
    )
    yield
    logger.info("Shutting down FraudScope.")


app = FastAPI(
    title="FraudScope",
    description=(
        "ML-powered transaction risk scoring service. Trained on a public "
        "or synthetic-for-demo dataset -- not connected to any real bank "
        "or customer data. See README for the full honesty statement."
    ),
    version=settings.model_version,
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(score.router, tags=["scoring"])
app.include_router(retrain.router, tags=["retrain (demo stub)"])
