import json

import pytest

from app.core.model_registry import ModelRegistry


def test_missing_model_file_is_reported(settings_factory):
    registry = ModelRegistry(settings_factory())
    with pytest.raises(RuntimeError, match="Komponen model tidak tersedia"):
        registry.load()


def test_feature_shape_contract(monkeypatch, settings_factory):
    settings = settings_factory(all_files=True)
    settings.model_metadata_path.write_text(json.dumps({
        "feature_shape": 21250, "label_map": {"0": "REAL", "1": "FAKE"}, "correction_config": {},
    }), encoding="utf-8")
    registry = ModelRegistry(settings)
    from app.core import model_registry as module
    module.pipeline.CLASSIFIER_MODEL = type("Model", (), {"n_features_in_": 7})()
    module.pipeline.V21_CONFIG = {}
    monkeypatch.setattr(module.pipeline, "get_threshold", lambda: 0.5)
    with pytest.raises(RuntimeError, match="Feature shape"):
        registry._validate_contract()

