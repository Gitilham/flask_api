def test_health_is_liveness_only(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.headers["X-Request-ID"]


def test_ready_when_registry_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_models_status_has_safe_contract(client):
    payload = client.get("/models/status").json()
    assert payload["model_loaded"] is True
    assert payload["label_map"] == {"0": "REAL", "1": "FAKE"}
    assert "paths" not in payload

