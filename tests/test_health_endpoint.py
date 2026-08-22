def test_health_returns_200_and_expected_keys(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_version" in body
    assert "uptime_s" in body
    assert body["uptime_s"] >= 0
