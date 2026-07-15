import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings


@pytest.fixture
def settings_factory(tmp_path):
    def factory(all_files=False):
        settings = Settings(
            config_path=tmp_path / "config.json", model_path=tmp_path / "model.pkl",
            scaler_path=tmp_path / "scaler.pkl", yolo_path=tmp_path / "face.pt",
            xception_encoder_path=tmp_path / "encoder.keras", model_metadata_path=tmp_path / "metadata.json",
            temp_folder=tmp_path / "temp",
        )
        if all_files:
            for path in [*settings.required_model_files.values(), settings.model_metadata_path]:
                path.touch()
        return settings
    return factory


@pytest.fixture
def client(monkeypatch, tmp_path):
    import main

    def fake_load(registry):
        registry.loaded = True
        registry.error = None
        registry.file_availability = {name: True for name in registry.settings.required_model_files}
        registry.load_duration_seconds = 0.01

    monkeypatch.setattr(main.ModelRegistry, "load", fake_load)
    monkeypatch.setattr(main.ModelRegistry, "unload", lambda registry: None)
    main.settings.temp_folder = tmp_path / "temp"
    main.settings.preserve_uploads = False
    with TestClient(main.app) as test_client:
        yield test_client

