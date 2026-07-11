import os
import json
import time
import uuid
import traceback
from typing import List, Tuple, Dict, Any

import cv2
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# TensorFlow dan YOLO dipakai untuk encoder frame dan deteksi wajah.
import tensorflow as tf
from ultralytics import YOLO


load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
TEMP_FOLDER = os.getenv("TEMP_FOLDER", "temp")
MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "300"))

CONFIG_PATH = os.getenv("CONFIG_PATH", "models/model_config_v3_fixed.json")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_v3_artifact_xception_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/feature_scaler.pkl")
YOLO_PATH = os.getenv("YOLO_PATH", "models/face_yolov8n.pt")
XCEPTION_ENCODER_PATH = os.getenv("XCEPTION_ENCODER_PATH", "models/xception_frame_encoder_safe.keras")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

CONFIG: Dict[str, Any] = {}
CLASSIFIER_PAYLOAD = None
CLASSIFIER_MODEL = None
SCALER = None
YOLO_MODEL = None
XCEPTION_ENCODER = None

MODEL_READY = False
MODEL_ERROR = None


def allowed_file(filename: str) -> bool:
    """
    Mengecek ekstensi file video.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_json_config() -> Dict[str, Any]:
    """
    Membaca file konfigurasi model hasil export dari Google Colab.
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"File config tidak ditemukan: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_all_models() -> None:
    """
    Load semua komponen model:
    1. Config model
    2. Classifier sklearn ExtraTrees dari file .pkl
    3. Scaler artifact features
    4. YOLOv8 face detector
    5. Xception frame encoder .keras

    Fungsi ini dijalankan sekali saat Flask start.
    """
    global CONFIG
    global CLASSIFIER_PAYLOAD
    global CLASSIFIER_MODEL
    global SCALER
    global YOLO_MODEL
    global XCEPTION_ENCODER
    global MODEL_READY
    global MODEL_ERROR

    try:
        CONFIG = load_json_config()

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"File classifier tidak ditemukan: {MODEL_PATH}")

        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"File scaler tidak ditemukan: {SCALER_PATH}")

        if not os.path.exists(YOLO_PATH):
            raise FileNotFoundError(f"File YOLO tidak ditemukan: {YOLO_PATH}")

        if not os.path.exists(XCEPTION_ENCODER_PATH):
            raise FileNotFoundError(f"File Xception encoder tidak ditemukan: {XCEPTION_ENCODER_PATH}")

        print("[INFO] Loading classifier:", MODEL_PATH)
        CLASSIFIER_PAYLOAD = joblib.load(MODEL_PATH)

        # File .pkl yang kamu punya berisi dictionary.
        # Model utama berada pada key best_single_model.
        if isinstance(CLASSIFIER_PAYLOAD, dict):
            CLASSIFIER_MODEL = CLASSIFIER_PAYLOAD.get("best_single_model")

            if CLASSIFIER_MODEL is None:
                final_model_name = CLASSIFIER_PAYLOAD.get("final_model_name")
                trained_models = CLASSIFIER_PAYLOAD.get("trained_models", {})

                if final_model_name and final_model_name in trained_models:
                    CLASSIFIER_MODEL = trained_models[final_model_name]
        else:
            CLASSIFIER_MODEL = CLASSIFIER_PAYLOAD

        if CLASSIFIER_MODEL is None:
            raise RuntimeError("Classifier utama tidak ditemukan di file .pkl.")

        print("[INFO] Loading scaler:", SCALER_PATH)
        SCALER = joblib.load(SCALER_PATH)

        print("[INFO] Loading YOLO:", YOLO_PATH)
        YOLO_MODEL = YOLO(YOLO_PATH)

        print("[INFO] Loading Xception encoder:", XCEPTION_ENCODER_PATH)
        XCEPTION_ENCODER = tf.keras.models.load_model(XCEPTION_ENCODER_PATH, compile=False)

        MODEL_READY = True
        MODEL_ERROR = None

        print("[INFO] Semua model berhasil dimuat.")
        print("[INFO] Classifier:", type(CLASSIFIER_MODEL))
        print("[INFO] Expected classifier features:", getattr(CLASSIFIER_MODEL, "n_features_in_", "unknown"))
        print("[INFO] Scaler features:", getattr(SCALER, "n_features_in_", "unknown"))
        print("[INFO] Threshold:", get_threshold())

    except Exception as error:
        MODEL_READY = False
        MODEL_ERROR = str(error)
        print("[ERROR] Gagal load model:")
        traceback.print_exc()


def get_threshold() -> float:
    """
    Mengambil threshold dari config/model payload.
    Pada model kamu threshold-nya adalah 0.4700000000000001.
    """
    if isinstance(CLASSIFIER_PAYLOAD, dict) and "threshold" in CLASSIFIER_PAYLOAD:
        return float(CLASSIFIER_PAYLOAD["threshold"])

    if CONFIG and "threshold" in CONFIG:
        return float(CONFIG["threshold"])

    return 0.5


def get_seq_len() -> int:
    return int(CONFIG.get("seq_len_use", 24))


def get_sample_fps() -> int:
    return int(CONFIG.get("sample_fps", 2))


def get_max_video_seconds() -> int:
    return int(CONFIG.get("max_video_seconds", 20))


def get_xception_img_size() -> int:
    return int(CONFIG.get("xception_img_size", 160))


def get_yolo_conf() -> float:
    return float(CONFIG.get("yolo_conf", 0.35))


def get_face_pad_ratio() -> float:
    return float(CONFIG.get("face_pad_ratio", 0.22))


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def read_video_frames(video_path: str, seq_len: int) -> List[Tuple[float, np.ndarray]]:
    """
    Mengambil frame video secara merata sampai jumlah seq_len.
    Output:
    [
        (frame_time_seconds, frame_bgr),
        ...
    ]
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Video tidak dapat dibuka oleh OpenCV.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None or fps <= 0:
        fps = 25.0

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Frame video tidak terbaca.")

    max_seconds = get_max_video_seconds()
    max_frame = min(total_frames, int(max_seconds * fps))

    if max_frame <= 0:
        max_frame = total_frames

    frame_indices = np.linspace(0, max_frame - 1, seq_len).astype(int)

    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()

        if not success or frame is None:
            continue

        frame_time = round(float(idx) / float(fps), 2)
        frames.append((frame_time, frame))

    cap.release()

    if not frames:
        raise RuntimeError("Tidak ada frame yang berhasil diekstrak dari video.")

    # Jika frame kurang dari seq_len, ulang frame terakhir.
    while len(frames) < seq_len:
        frames.append(frames[-1])

    return frames[:seq_len]


def crop_face_with_yolo(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Deteksi wajah menggunakan YOLOv8.
    Jika wajah ditemukan, ambil bounding box terbesar/terpercaya.
    Jika tidak ditemukan, gunakan center crop agar sistem tetap bisa memproses.
    """
    h, w = frame_bgr.shape[:2]

    try:
        results = YOLO_MODEL.predict(
            source=frame_bgr,
            conf=get_yolo_conf(),
            verbose=False
        )

        best_box = None
        best_score = -1

        if results and len(results) > 0:
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0

                    x1, y1, x2, y2 = xyxy
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    score = area * (conf + 0.01)

                    if score > best_score:
                        best_score = score
                        best_box = (int(x1), int(y1), int(x2), int(y2))

        if best_box is not None:
            x1, y1, x2, y2 = best_box

            bw = x2 - x1
            bh = y2 - y1
            pad = int(max(bw, bh) * get_face_pad_ratio())

            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = frame_bgr[y1:y2, x1:x2]

            if crop.size > 0:
                return crop

    except Exception:
        # Jika YOLO gagal pada frame tertentu, pakai center crop.
        pass

    # Fallback center crop
    size = min(h, w)
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    return frame_bgr[start_y:start_y + size, start_x:start_x + size]


def prepare_face_for_xception(face_bgr: np.ndarray) -> np.ndarray:
    """
    Resize crop wajah untuk input Xception encoder.
    Model encoder sudah memiliki preprocessing/rescaling layer,
    jadi input dibuat float32 range 0-255.
    """
    img_size = get_xception_img_size()

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    face_rgb = face_rgb.astype("float32")

    return face_rgb


def extract_artifact_features_from_frame(face_bgr: np.ndarray) -> np.ndarray:
    """
    Ekstraksi fitur sederhana dari satu frame/crop wajah.
    Fitur ini dipakai untuk membuat 20 artifact features video.
    """
    resized = cv2.resize(face_bgr, (160, 160), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 80, 160)

    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    contrast = float(np.std(gray))

    blur_var = float(np.var(lap))
    lap_mean = float(np.mean(np.abs(lap)))

    sat_mean = float(np.mean(hsv[:, :, 1]))
    sat_std = float(np.std(hsv[:, :, 1]))

    val_mean = float(np.mean(hsv[:, :, 2]))
    val_std = float(np.std(hsv[:, :, 2]))

    edge_density = float(np.mean(edges > 0))

    return np.array([
        brightness_mean,
        brightness_std,
        contrast,
        blur_var,
        lap_mean,
        sat_mean,
        sat_std,
        val_mean,
        val_std,
        edge_density,
    ], dtype=np.float32)


def extract_video_artifact_features(face_crops_bgr: List[np.ndarray]) -> np.ndarray:
    """
    Membuat 20 fitur artifact video.
    Caranya:
    - Ambil 10 fitur per frame
    - Hitung mean dan std antar frame
    - 10 mean + 10 std = 20 fitur
    """
    per_frame_features = []

    for crop in face_crops_bgr:
        per_frame_features.append(extract_artifact_features_from_frame(crop))

    arr = np.array(per_frame_features, dtype=np.float32)

    mean_features = np.mean(arr, axis=0)
    std_features = np.std(arr, axis=0)

    artifact_20 = np.concatenate([mean_features, std_features], axis=0).astype(np.float32)

    scaler_features = int(getattr(SCALER, "n_features_in_", 20))

    if artifact_20.shape[0] < scaler_features:
        artifact_20 = np.pad(artifact_20, (0, scaler_features - artifact_20.shape[0]))

    if artifact_20.shape[0] > scaler_features:
        artifact_20 = artifact_20[:scaler_features]

    artifact_scaled = SCALER.transform([artifact_20])[0]

    return artifact_scaled.astype(np.float32)


def build_temporal_xception_features(embeddings: np.ndarray, target_len: int) -> np.ndarray:
    """
    Membuat fitur Xception video dari embedding frame.

    Karena classifier dari training mengharapkan jumlah fitur tertentu,
    fungsi ini membuat gabungan fitur temporal lalu menyesuaikan panjangnya
    agar cocok dengan n_features_in_ classifier.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        embeddings = embeddings.reshape((embeddings.shape[0], -1))

    # Fitur temporal umum
    mean_feat = np.mean(embeddings, axis=0)
    std_feat = np.std(embeddings, axis=0)
    min_feat = np.min(embeddings, axis=0)
    max_feat = np.max(embeddings, axis=0)
    median_feat = np.median(embeddings, axis=0)

    q25_feat = np.percentile(embeddings, 25, axis=0)
    q75_feat = np.percentile(embeddings, 75, axis=0)

    if embeddings.shape[0] > 1:
        diff = np.diff(embeddings, axis=0)
        diff_mean = np.mean(diff, axis=0)
        diff_std = np.std(diff, axis=0)
    else:
        diff_mean = np.zeros_like(mean_feat)
        diff_std = np.zeros_like(std_feat)

    flat_seq = embeddings.flatten()

    combined = np.concatenate([
        mean_feat,
        std_feat,
        min_feat,
        max_feat,
        median_feat,
        q25_feat,
        q75_feat,
        diff_mean,
        diff_std,
        flat_seq,
    ], axis=0).astype(np.float32)

    if combined.shape[0] < target_len:
        combined = np.pad(combined, (0, target_len - combined.shape[0]))

    if combined.shape[0] > target_len:
        combined = combined[:target_len]

    return combined.astype(np.float32)


def extract_features_from_video(video_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Proses utama ekstraksi fitur:
    1. Ambil 24 frame dari video
    2. Deteksi/crop wajah dengan YOLO
    3. Encode crop wajah dengan Xception encoder
    4. Ekstraksi artifact features
    5. Gabungkan semua fitur agar cocok dengan classifier
    """
    seq_len = get_seq_len()
    frames = read_video_frames(video_path, seq_len)

    face_crops_bgr = []
    xception_inputs = []
    frame_infos = []

    for frame_time, frame_bgr in frames:
        face_crop = crop_face_with_yolo(frame_bgr)
        face_crops_bgr.append(face_crop)

        x_input = prepare_face_for_xception(face_crop)
        xception_inputs.append(x_input)

        frame_infos.append({
            "frame_time": frame_time
        })

    xception_batch = np.array(xception_inputs, dtype=np.float32)

    embeddings = XCEPTION_ENCODER.predict(xception_batch, verbose=0)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    expected_total_features = int(getattr(CLASSIFIER_MODEL, "n_features_in_", 21250))
    artifact_features = extract_video_artifact_features(face_crops_bgr)

    deep_target_len = expected_total_features - artifact_features.shape[0]

    if deep_target_len <= 0:
        raise RuntimeError("Jumlah fitur classifier tidak valid.")

    deep_features = build_temporal_xception_features(embeddings, deep_target_len)

    # Urutan fitur: fitur Xception temporal + artifact features.
    final_features = np.concatenate([deep_features, artifact_features], axis=0).astype(np.float32)

    if final_features.shape[0] < expected_total_features:
        final_features = np.pad(final_features, (0, expected_total_features - final_features.shape[0]))

    if final_features.shape[0] > expected_total_features:
        final_features = final_features[:expected_total_features]

    return final_features.reshape(1, -1), frame_infos


def predict_with_classifier(features: np.ndarray) -> Dict[str, Any]:
    """
    Prediksi REAL/DEEPFAKE menggunakan classifier sklearn.

    Catatan penting:
    Config model kamu menulis:
    FIXED_NO_FLIP. Jangan membalik score pada backend Flask.

    Jadi score tidak dibalik.
    Class 0 = REAL
    Class 1 = FAKE/DEEPFAKE
    """
    threshold = get_threshold()

    if hasattr(CLASSIFIER_MODEL, "predict_proba"):
        proba = CLASSIFIER_MODEL.predict_proba(features)[0]

        classes = list(CLASSIFIER_MODEL.classes_)

        if 0 in classes:
            real_score = float(proba[classes.index(0)])
        else:
            real_score = float(proba[0])

        if 1 in classes:
            fake_score = float(proba[classes.index(1)])
        else:
            fake_score = float(proba[-1])

    else:
        pred = int(CLASSIFIER_MODEL.predict(features)[0])

        if pred == 1:
            real_score = 0.0
            fake_score = 1.0
        else:
            real_score = 1.0
            fake_score = 0.0

    if fake_score >= threshold:
        prediction = "DEEPFAKE"
        confidence = fake_score
    else:
        prediction = "REAL"
        confidence = real_score

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 6),
        "real_score": round(float(real_score), 6),
        "fake_score": round(float(fake_score), 6),
        "threshold": threshold
    }


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "success": True,
        "message": "Flask API Deepfake Detection aktif",
        "endpoint": "/predict-video",
        "model_ready": MODEL_READY,
        "model_error": MODEL_ERROR,
        "model_name": CONFIG.get("model_name") if CONFIG else None
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": MODEL_READY,
        "status": "healthy" if MODEL_READY else "model_error",
        "message": "API berjalan normal" if MODEL_READY else "API aktif tetapi model gagal dimuat",
        "model_error": MODEL_ERROR
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    info = {
        "success": MODEL_READY,
        "model_ready": MODEL_READY,
        "model_error": MODEL_ERROR,
        "config": CONFIG,
        "paths": {
            "config_path": CONFIG_PATH,
            "model_path": MODEL_PATH,
            "scaler_path": SCALER_PATH,
            "yolo_path": YOLO_PATH,
            "xception_encoder_path": XCEPTION_ENCODER_PATH
        }
    }

    if CLASSIFIER_MODEL is not None:
        info["classifier_type"] = str(type(CLASSIFIER_MODEL))
        info["classifier_features"] = int(getattr(CLASSIFIER_MODEL, "n_features_in_", 0))
        info["classifier_classes"] = [int(x) for x in getattr(CLASSIFIER_MODEL, "classes_", [])]

    if SCALER is not None:
        info["scaler_features"] = int(getattr(SCALER, "n_features_in_", 0))

    return jsonify(info)


@app.route("/predict-video", methods=["POST"])
def predict_video():
    start_time = time.time()

    if not MODEL_READY:
        return jsonify({
            "success": False,
            "message": "Model belum siap atau gagal dimuat.",
            "model_error": MODEL_ERROR
        }), 500

    if "video" not in request.files:
        return jsonify({
            "success": False,
            "message": "File video tidak ditemukan. Gunakan field bernama video."
        }), 400

    video = request.files["video"]

    if video.filename == "":
        return jsonify({
            "success": False,
            "message": "Nama file video kosong."
        }), 400

    if not allowed_file(video.filename):
        return jsonify({
            "success": False,
            "message": "Format video tidak didukung. Gunakan mp4, avi, mov, atau mkv."
        }), 400

    original_filename = secure_filename(video.filename)
    ext = original_filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

    try:
        video.save(save_path)

        features, frame_infos = extract_features_from_video(save_path)
        result = predict_with_classifier(features)

        duration_seconds = round(time.time() - start_time, 2)

        # Detail frame sederhana untuk ditampilkan di website.
        frames_response = []
        for frame in frame_infos:
            frames_response.append({
                "frame_time": frame["frame_time"],
                "label": result["prediction"],
                "confidence": result["confidence"],
                "real_score": result["real_score"],
                "fake_score": result["fake_score"]
            })

        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "real_score": result["real_score"],
            "fake_score": result["fake_score"],
            "threshold": result["threshold"],
            "duration_seconds": duration_seconds,
            "message": "Prediksi berhasil",
            "frames": frames_response
        })

    except Exception as error:
        traceback.print_exc()

        return jsonify({
            "success": False,
            "message": f"Video tidak dapat diproses: {str(error)}"
        }), 500


if __name__ == "__main__":
    load_all_models()

    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    app.run(host=host, port=port, debug=debug)