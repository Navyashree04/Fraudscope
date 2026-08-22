from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_payload():
    return {
        "transaction_id": "txn_test_1",
        "Time": 40000.0,
        "Amount": 149.62,
        **{f"V{i}": 0.0 for i in range(1, 29)},
    }
