def test_valid_payload_returns_200_with_correct_schema(client, sample_payload):
    r = client.post("/score", json=sample_payload)
    assert r.status_code == 200
    body = r.json()

    assert body["transaction_id"] == sample_payload["transaction_id"]
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert len(body["top_features"]) == 3
    for f in body["top_features"]:
        assert set(f.keys()) == {"feature", "contribution", "direction"}
        assert f["direction"] in {"increases risk", "decreases risk"}
    assert "model_version" in body
    assert "scored_at" in body
    assert "latency_ms" in body


def test_missing_field_returns_422(client, sample_payload):
    bad = dict(sample_payload)
    del bad["V1"]
    r = client.post("/score", json=bad)
    assert r.status_code == 422


def test_wrong_type_returns_422(client, sample_payload):
    bad = dict(sample_payload)
    bad["Amount"] = "not-a-number"
    r = client.post("/score", json=bad)
    assert r.status_code == 422


def test_negative_amount_returns_422(client, sample_payload):
    bad = dict(sample_payload)
    bad["Amount"] = -10.0
    r = client.post("/score", json=bad)
    assert r.status_code == 422


def test_transaction_id_is_optional(client, sample_payload):
    payload = dict(sample_payload)
    del payload["transaction_id"]
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    assert r.json()["transaction_id"] is None
