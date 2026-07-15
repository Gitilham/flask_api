from pathlib import Path

import numpy as np
import pytest

from app.exceptions.handlers import ApiError
from app.services import prediction_service


def test_corrupt_video_is_rejected(monkeypatch, tmp_path):
    class ClosedCapture:
        def isOpened(self): return False
        def release(self): pass
    monkeypatch.setattr(prediction_service.cv2, "VideoCapture", lambda path: ClosedCapture())
    with pytest.raises(ApiError) as error:
        prediction_service.predict_video(tmp_path / "broken.mp4")
    assert error.value.error_code == "CORRUPT_VIDEO"


def test_no_face_response_keeps_legacy_contract(monkeypatch, tmp_path):
    class Capture:
        def isOpened(self): return True
        def read(self): return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass
    monkeypatch.setattr(prediction_service.cv2, "VideoCapture", lambda path: Capture())
    frame = {"frame_time": 0.0, "face_detected": False, "face_confidence": None,
             "crop_method": "center_crop", "repeated_frame": False, "bbox": [0, 0, 4, 4]}
    monkeypatch.setattr(prediction_service.pipeline, "extract_features_from_video", lambda path: (
        np.zeros((1, 21250), dtype=np.float32), [frame], {"face_detected_count": 0}
    ))
    monkeypatch.setattr(prediction_service.pipeline, "get_min_face_frames", lambda: 3)
    monkeypatch.setattr(prediction_service.pipeline, "get_threshold", lambda: 0.5)
    result = prediction_service.predict_video(tmp_path / "video.mp4")
    assert result["success"] is True
    assert result["prediction"] == result["label"] == "NO_FACE"
    assert result["threshold"] == 0.5

