import time
from pathlib import Path
from typing import Any

import cv2

from app.exceptions.handlers import ApiError
from app.services import v21_pipeline as pipeline


def _validate_video(path: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ApiError("Video tidak dapat dibuka atau rusak.", "CORRUPT_VIDEO", 422)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ApiError("Tidak ada frame yang dapat dibaca dari video.", "NO_FRAMES", 422)
    finally:
        capture.release()


def predict_video(path: Path) -> dict[str, Any]:
    started = time.time()
    probe_started = time.perf_counter()
    _validate_video(path)
    video_probe_seconds = time.perf_counter() - probe_started
    features, frame_infos, feature_debug = pipeline.extract_features_from_video(str(path))
    min_face_frames = pipeline.get_min_face_frames()
    face_count = int(feature_debug.get("face_detected_count", 0))

    if face_count < min_face_frames:
        frames = [{
            "frame_time": frame["frame_time"],
            "status": "wajah tidak terdeteksi" if not frame["face_detected"] else "wajah terdeteksi",
            "face_detected": frame["face_detected"], "face_confidence": frame["face_confidence"],
            "crop_method": frame["crop_method"], "repeated_frame": frame["repeated_frame"],
            "bbox": frame["bbox"],
            "note": "Video tidak diklasifikasikan karena jumlah frame wajah tidak mencukupi.",
        } for frame in frame_infos]
        return {
            "success": True, "prediction": "NO_FACE", "label": "NO_FACE", "confidence": 0.0,
            "result": "NO_FACE", "final_decision": "NO_FACE",
            "real_score": None, "fake_score": None, "threshold": pipeline.get_threshold(), "margin": None,
            "confidence_note": "Wajah tidak terdeteksi / frame wajah tidak mencukupi",
            "decision_rule": "Klasifikasi hanya dilakukan jika wajah terdeteksi minimal pada beberapa frame.",
            "decision_explanation": f"Video tidak diklasifikasikan karena hanya {face_count} frame wajah terdeteksi dari minimal {min_face_frames} frame yang dibutuhkan.",
            "duration_seconds": round(time.time() - started, 2), "processing_seconds": round(time.time() - started, 2),
            "message": "Wajah tidak terdeteksi atau tidak cukup jelas. Upload video wajah untuk dianalisis.",
            "frames_used": len(frame_infos), "face_detected_count": face_count,
            "min_face_frames": min_face_frames, "feature_debug": feature_debug, "frames": frames,
        }

    classifier_started = time.perf_counter()
    result = pipeline.predict_with_classifier(features)
    classifier_seconds = time.perf_counter() - classifier_started
    feature_debug.setdefault("timings", {}).update({
        "video_probe_seconds": round(video_probe_seconds, 6),
        "classifier_and_local_similarity_seconds": round(classifier_seconds, 6),
    })
    frames = [{
        "frame_time": frame["frame_time"], "status": "frame digunakan",
        "face_detected": frame["face_detected"], "face_confidence": frame["face_confidence"],
        "crop_method": frame["crop_method"], "repeated_frame": frame["repeated_frame"],
        "bbox": frame["bbox"], "note": "Score prediksi dihitung pada level video, bukan per-frame.",
    } for frame in frame_infos]
    return {
        "success": True, "prediction": result["prediction"],
        "label": result.get("label", result["prediction"]),
        "result": result.get("label", result["prediction"]),
        "final_decision": result.get("label", result["prediction"]),
        "status": result.get("status", result.get("label", result["prediction"])),
        "confidence": result["confidence"], "real_score": result["real_score"],
        "fake_score": result["fake_score"], "base_score_fake": result.get("base_score_fake"),
        "local_score_fake": result.get("local_score_fake"), "threshold": result["threshold"],
        "margin": result["margin"], "confidence_note": result["confidence_note"],
        "decision_rule": result["decision_rule"], "decision_explanation": result["decision_explanation"],
        "model_version": result.get("model_version", pipeline.MODEL_VERSION),
        "duration_seconds": round(time.time() - started, 2), "processing_seconds": round(time.time() - started, 2),
        "message": "Prediksi berhasil", "frames_used": len(frame_infos), "faces": int(feature_debug.get("face_detected_count", 0)),
        "feature_debug": feature_debug, "frames": frames,
    }
