import ast
from pathlib import Path

import numpy as np

from app.services import v21_pipeline

ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_FUNCTIONS = {
    "read_video_frames", "prepare_face_for_xception", "aggregate_sequence_stats",
    "extract_basic_visual_features_from_frame", "make_lbp_histogram", "blockiness_features",
    "dct_frequency_features", "extract_rich_artifact_features_from_frame", "build_v3_feature_vector",
    "extract_features_from_video", "l2_normalize", "sigmoid", "logit", "get_model_fake_score",
    "apply_flip_score", "knn_prob_fake_single", "blend_prob",
}


def _function_sources(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    return {node.name: ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_migrated_algorithm_is_textually_identical_to_flask_backup():
    legacy = _function_sources(ROOT / "legacy" / "app_flask_v21.py")
    migrated = _function_sources(ROOT / "app" / "services" / "v21_pipeline.py")
    for name in ALGORITHM_FUNCTIONS:
        assert migrated[name] == legacy[name], name


def test_v21_probability_math_contract(monkeypatch):
    class FakeModel:
        classes_ = np.array([0, 1])
        n_features_in_ = 3
        def predict_proba(self, values):
            return np.array([[0.45, 0.55]], dtype=np.float64)

    v21_pipeline.CLASSIFIER_PAYLOAD = {"base_model": FakeModel()}
    v21_pipeline.CLASSIFIER_MODEL = FakeModel()
    v21_pipeline.V21_BASE_USE_FLIP = False
    v21_pipeline.V21_X_REF_NORM = v21_pipeline.l2_normalize(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    v21_pipeline.V21_Y_REF = np.array([0, 1], dtype=np.int64)
    v21_pipeline.V21_CONFIG = {"k": 2, "temp": 0.06, "power": 1.0, "alpha": 0.55, "mode": "borderline"}
    monkeypatch.setattr(v21_pipeline, "get_threshold", lambda: 0.5)
    result = v21_pipeline.predict_with_v21(np.array([[0, 1, 0]], dtype=np.float32))
    assert result["threshold"] == 0.5
    assert result["prediction"] in {"REAL", "DEEPFAKE"}


def test_exactly_fifty_percent_is_suspicious(monkeypatch):
    class FakeModel:
        classes_ = np.array([0, 1])
        n_features_in_ = 3
        def predict_proba(self, values):
            return np.array([[0.5, 0.5]], dtype=np.float64)

    v21_pipeline.CLASSIFIER_PAYLOAD = {"base_model": FakeModel()}
    v21_pipeline.CLASSIFIER_MODEL = FakeModel()
    v21_pipeline.V21_BASE_USE_FLIP = False
    v21_pipeline.V21_X_REF_NORM = v21_pipeline.l2_normalize(np.array([[1, 0, 0]], dtype=np.float32))
    v21_pipeline.V21_Y_REF = np.array([1], dtype=np.int64)
    v21_pipeline.V21_CONFIG = {"k": 1, "temp": 0.06, "power": 1.0, "alpha": 1.0, "mode": "linear"}
    monkeypatch.setattr(v21_pipeline, "get_threshold", lambda: 0.5)
    result = v21_pipeline.predict_with_v21(np.array([[1, 0, 0]], dtype=np.float32))
    assert result["fake_score"] == 0.5
    assert result["prediction"] == result["label"] == result["status"] == "MENCURIGAKAN"
