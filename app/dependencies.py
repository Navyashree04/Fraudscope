"""
Model + explainer singleton loader.

The trained model artifact is loaded ONCE at process startup (via FastAPI's
lifespan hook in app/main.py), never re-trained or reloaded per-request.
Reminder: the loaded artifact was trained only on public/synthetic data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import joblib

from src.config import settings
from src.explain import Explainer


@dataclass
class ModelState:
    model: object
    explainer: Explainer
    model_version: str
    metadata: dict
    started_at: float


_state: Optional[ModelState] = None


def load_model_state() -> ModelState:
    global _state
    model = joblib.load(settings.model_path)
    with open(settings.model_metadata_path) as f:
        metadata = json.load(f)
    explainer = Explainer(model)
    _state = ModelState(
        model=model,
        explainer=explainer,
        model_version=metadata.get("model_version", settings.model_version),
        metadata=metadata,
        started_at=time.time(),
    )
    return _state


def get_model_state() -> ModelState:
    if _state is None:
        raise RuntimeError(
            "Model state not loaded yet -- this should be set during the "
            "FastAPI lifespan startup hook."
        )
    return _state
