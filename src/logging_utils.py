"""
Structured JSON request/score logging -- a drift-monitoring stub.

In a real production fraud system, this log stream would feed a
monitoring pipeline (e.g. Evidently, a feature-store drift dashboard, or a
custom data-quality job) that watches for input feature drift and score
distribution shifts over time. Here, it simply writes structured JSON
lines to the standard logger so the *schema* is demonstrated even though
no actual monitoring backend is wired up.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

logger = logging.getLogger("fraudscope.scoring")


def log_score_event(
    transaction_id: Optional[str],
    feature_summary: dict,
    risk_score: float,
    risk_band: str,
    model_version: str,
    latency_ms: float,
) -> None:
    """Emit one structured JSON log line per scored transaction.

    Keeping this schema stable is what would let a real drift-monitoring
    pipeline consume it reliably -- adding/removing fields here should be
    treated like an API/schema change.
    """
    event = {
        "event": "fraudscope.score",
        "ts": time.time(),
        "transaction_id": transaction_id,
        "model_version": model_version,
        "risk_score": round(risk_score, 6),
        "risk_band": risk_band,
        "latency_ms": round(latency_ms, 3),
        # Summary only (min/max/mean of Amount-like fields), never the full
        # raw feature vector, to keep log volume and payload copies minimal.
        "feature_summary": feature_summary,
    }
    logger.info(json.dumps(event))
