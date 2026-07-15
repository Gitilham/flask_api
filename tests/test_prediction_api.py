def test_missing_upload_field(client):
    response = client.post("/predict-video")
    assert response.status_code == 400
    assert response.json()["error_code"] == "VIDEO_REQUIRED"


def test_invalid_extension(client):
    response = client.post("/predict-video", files={"video": ("bad.txt", b"x", "video/mp4")})
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_EXTENSION"


def test_invalid_mime(client):
    response = client.post("/predict-video", files={"video": ("bad.mp4", b"x", "text/plain")})
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_MIME"


def test_empty_upload_is_rejected_and_cleaned(client):
    response = client.post("/predict-video", files={"video": ("empty.mp4", b"", "video/mp4")})
    assert response.status_code == 400
    assert response.json()["error_code"] == "EMPTY_FILE"
    assert list(client.app.state.model_registry.settings.temp_folder.glob("*")) == []


def test_compatible_prediction_response_and_cleanup(client, monkeypatch):
    expected = {
        "success": True, "prediction": "REAL", "label": "REAL", "status": "REAL",
        "confidence": 0.8, "real_score": 0.8, "fake_score": 0.2,
        "threshold": 0.5, "margin": 0.3, "message": "Prediksi berhasil",
        "frames_used": 32, "feature_debug": {"feature_vector_shape": [1, 21250]}, "frames": [],
    }
    monkeypatch.setattr("app.api.prediction.predict_video", lambda path: expected)
    response = client.post("/predict-video", files={"video": ("sample.mp4", b"not-decoded-in-mock", "video/mp4")})
    assert response.status_code == 200
    assert response.json() == expected
    assert list(client.app.state.model_registry.settings.temp_folder.glob("*")) == []


def test_internal_error_does_not_expose_traceback(client, monkeypatch):
    monkeypatch.setattr("app.api.prediction.predict_video", lambda path: (_ for _ in ()).throw(RuntimeError("secret detail")))
    response = client.post("/predict-video", files={"video": ("sample.mp4", b"x", "video/mp4")})
    assert response.status_code == 500
    assert "traceback" not in response.text.lower()
    assert "secret detail" not in response.text


def test_inference_queue_timeout(client):
    import asyncio
    client.app.state.inference_semaphore = asyncio.Semaphore(0)
    client.app.state.model_registry.settings.inference_queue_timeout = 0.01
    response = client.post("/predict-video", files={"video": ("sample.mp4", b"x", "video/mp4")})
    assert response.status_code == 503
    assert response.json()["error_code"] == "QUEUE_TIMEOUT"
    assert list(client.app.state.model_registry.settings.temp_folder.glob("*")) == []

