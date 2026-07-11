import os
import json
import time
import uuid
import traceback
from typing import List, Tuple, Dict, Any, Optional

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


#w ============================================================
# APP CONFIG
# ============================================================

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
TEMP_FOLDER = os.getenv("TEMP_FOLDER", "temp")
MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "300"))

CONFIG_PATH = os.getenv("CONFIG_PATH", "models/config.json")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_v21_manual_audit_local_similarity.pkl")
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

# ============================================================
# V21 LOCAL SIMILARITY CORRECTION GLOBALS
# ============================================================

MODEL_VERSION = "UNKNOWN"
V21_BASE_USE_FLIP = False
V21_X_REF_NORM = None
V21_Y_REF = None
V21_CONFIG: Dict[str, Any] = {}
V21_SUSPICIOUS_MIN = float(os.getenv("V21_SUSPICIOUS_MIN", "0.40"))


# ============================================================
# BASIC HELPERS
# ============================================================

def allowed_file(filename: str) -> bool:
    """Mengecek ekstensi file video."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_json_config() -> Dict[str, Any]:
    """Membaca file konfigurasi model hasil export dari Google Colab."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"File config tidak ditemukan: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_threshold() -> float:
    """
    Mengambil threshold dari model payload/config.
    Model V3 kamu memakai threshold sekitar 0.47.
    """
    if isinstance(CLASSIFIER_PAYLOAD, dict) and "threshold" in CLASSIFIER_PAYLOAD:
        return float(CLASSIFIER_PAYLOAD["threshold"])

    if CONFIG and "threshold" in CONFIG:
        return float(CONFIG["threshold"])

    return 0.5


def get_seq_len() -> int:
    # V3 lama memakai seq_len_use, sedangkan export Colab baru sering memakai seq_len.
    return int(CONFIG.get("seq_len_use", CONFIG.get("seq_len", 24)))


def get_min_face_frames() -> int:
    """
    Minimal jumlah frame yang harus memiliki wajah hasil deteksi YOLO.
    Jika kurang dari nilai ini, video dianggap tidak valid/tidak ada wajah
    dan tidak dilanjutkan ke klasifikasi deepfake.
    """
    return int(CONFIG.get("min_face_frames", os.getenv("MIN_FACE_FRAMES", 3)))


def get_max_video_seconds() -> int:
    return int(CONFIG.get("max_video_seconds", 20))


def get_xception_img_size() -> int:
    # Jangan otomatis memakai img_size dari config training jika encoder backend lama inputnya 160.
    # Set XCEPTION_IMG_SIZE di .env kalau encoder kamu memang berbeda.
    return int(os.getenv("XCEPTION_IMG_SIZE", CONFIG.get("xception_img_size", 160)))


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


def pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Menyesuaikan panjang fitur tanpa mengubah urutan fitur yang sudah dibuat.
    Dipakai hanya sebagai pengaman agar panjang fitur sama dengan training.
    """
    arr = np.asarray(arr, dtype=np.float32).ravel()

    if target_len <= 0:
        return np.array([], dtype=np.float32)

    if arr.shape[0] < target_len:
        arr = np.pad(arr, (0, target_len - arr.shape[0]))

    if arr.shape[0] > target_len:
        arr = arr[:target_len]

    return arr.astype(np.float32)


# ============================================================
# MODEL LOADING
# ============================================================

def load_all_models() -> None:
    """
    Load semua komponen model:
    1. Config model
    2. Classifier sklearn / payload V21 dari file .pkl
    3. Scaler fitur visual dasar
    4. YOLOv8 face detector
    5. Xception frame encoder .keras

    Catatan:
    - Untuk V21, file .pkl berisi base_model + local similarity correction.
    - CLASSIFIER_MODEL tetap diisi base_model agar extractor tahu jumlah fitur target.
    """
    global CONFIG
    global CLASSIFIER_PAYLOAD
    global CLASSIFIER_MODEL
    global SCALER
    global YOLO_MODEL
    global XCEPTION_ENCODER
    global MODEL_READY
    global MODEL_ERROR
    global MODEL_VERSION
    global V21_BASE_USE_FLIP
    global V21_X_REF_NORM
    global V21_Y_REF
    global V21_CONFIG

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

        print("[INFO] Loading classifier/payload:", MODEL_PATH)
        CLASSIFIER_PAYLOAD = joblib.load(MODEL_PATH)

        MODEL_VERSION = "UNKNOWN"
        V21_BASE_USE_FLIP = False
        V21_X_REF_NORM = None
        V21_Y_REF = None
        V21_CONFIG = {}

        if isinstance(CLASSIFIER_PAYLOAD, dict):
            # ====================================================
            # FORMAT V21:
            # {
            #   base_model, base_use_flip, threshold,
            #   X_ref_norm, y_ref, correction_config, model_version
            # }
            # ====================================================
            if "base_model" in CLASSIFIER_PAYLOAD and "X_ref_norm" in CLASSIFIER_PAYLOAD and "y_ref" in CLASSIFIER_PAYLOAD:
                MODEL_VERSION = str(CLASSIFIER_PAYLOAD.get("model_version", "V21_LOCAL_SIMILARITY_CORRECTION"))
                CLASSIFIER_MODEL = CLASSIFIER_PAYLOAD.get("base_model")
                V21_BASE_USE_FLIP = bool(CLASSIFIER_PAYLOAD.get("base_use_flip", False))
                V21_X_REF_NORM = np.asarray(CLASSIFIER_PAYLOAD.get("X_ref_norm"), dtype=np.float32)
                V21_Y_REF = np.asarray(CLASSIFIER_PAYLOAD.get("y_ref"), dtype=np.int64)
                V21_CONFIG = dict(CLASSIFIER_PAYLOAD.get("correction_config", {}))

                if CLASSIFIER_MODEL is None:
                    raise RuntimeError("base_model tidak ditemukan di payload V21.")

                if V21_X_REF_NORM.ndim != 2 or V21_Y_REF.ndim != 1:
                    raise RuntimeError("X_ref_norm/y_ref pada payload V21 tidak valid.")

                print("[INFO] Detected V21 local similarity payload.")
                print("[INFO] V21 correction config:", V21_CONFIG)
                print("[INFO] V21 reference bank:", V21_X_REF_NORM.shape)

            else:
                # ====================================================
                # FORMAT V3 LAMA
                # ====================================================
                MODEL_VERSION = "V3_ARTIFACT_XCEPTION"
                CLASSIFIER_MODEL = CLASSIFIER_PAYLOAD.get("best_single_model")

                if CLASSIFIER_MODEL is None:
                    final_model_name = CLASSIFIER_PAYLOAD.get("final_model_name")
                    trained_models = CLASSIFIER_PAYLOAD.get("trained_models", {})

                    if final_model_name and final_model_name in trained_models:
                        CLASSIFIER_MODEL = trained_models[final_model_name]
        else:
            MODEL_VERSION = "SKLEARN_DIRECT"
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
        print("[INFO] Model version:", MODEL_VERSION)
        print("[INFO] Classifier:", type(CLASSIFIER_MODEL))
        print("[INFO] Expected classifier features:", getattr(CLASSIFIER_MODEL, "n_features_in_", "unknown"))
        print("[INFO] Classifier classes:", getattr(CLASSIFIER_MODEL, "classes_", "unknown"))
        print("[INFO] Scaler features:", getattr(SCALER, "n_features_in_", "unknown"))
        print("[INFO] Threshold:", get_threshold())
        print("[INFO] Seq len:", get_seq_len())
        print("[INFO] Xception image size:", get_xception_img_size())

    except Exception as error:
        MODEL_READY = False
        MODEL_ERROR = str(error)
        print("[ERROR] Gagal load model:")
        traceback.print_exc()

# ============================================================
# VIDEO READ + FACE DETECTION
# ============================================================

def read_video_frames(video_path: str, seq_len: int) -> List[Tuple[float, np.ndarray, bool]]:
    """
    Mengambil frame video secara merata sampai jumlah seq_len.

    Output:
    [
        (frame_time_seconds, frame_bgr, repeated_frame),
        ...
    ]

    repeated_frame = True jika frame tersebut hasil duplikasi frame terakhir,
    dipakai ketika frame valid kurang dari seq_len.
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

    frames: List[Tuple[float, np.ndarray, bool]] = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()

        if not success or frame is None:
            continue

        frame_time = round(float(idx) / float(fps), 2)
        frames.append((frame_time, frame, False))

    cap.release()

    if not frames:
        raise RuntimeError("Tidak ada frame yang berhasil diekstrak dari video.")

    # Jika frame kurang dari seq_len, ulang frame terakhir agar panjang sequence tetap.
    while len(frames) < seq_len:
        last_time, last_frame, _ = frames[-1]
        frames.append((last_time, last_frame.copy(), True))

    return frames[:seq_len]


def crop_face_with_yolo(frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Deteksi wajah menggunakan YOLOv8.
    Jika wajah ditemukan, ambil bounding box terbaik.
    Jika tidak ditemukan, gunakan center crop agar sistem tetap bisa memproses.

    Return:
    - crop wajah
    - metadata deteksi
    """
    h, w = frame_bgr.shape[:2]
    metadata = {
        "face_detected": False,
        "face_confidence": None,
        "bbox": None,
        "crop_method": "center_crop"
    }

    try:
        results = YOLO_MODEL.predict(
            source=frame_bgr,
            conf=get_yolo_conf(),
            verbose=False
        )

        best_box = None
        best_conf = 0.0
        best_score = -1.0

        if results and len(results) > 0:
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0

                    x1, y1, x2, y2 = xyxy
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    score = area * (conf + 0.01)

                    if score > best_score:
                        best_score = score
                        best_conf = conf
                        best_box = (int(x1), int(y1), int(x2), int(y2))

        if best_box is not None:
            x1, y1, x2, y2 = best_box

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            pad = int(max(bw, bh) * get_face_pad_ratio())

            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)

            crop = frame_bgr[y1p:y2p, x1p:x2p]

            if crop.size > 0:
                metadata.update({
                    "face_detected": True,
                    "face_confidence": round(float(best_conf), 6),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_padded": [int(x1p), int(y1p), int(x2p), int(y2p)],
                    "crop_method": "yolo"
                })
                return crop, metadata

    except Exception:
        # Jika YOLO gagal pada frame tertentu, pakai center crop.
        pass

    # Fallback center crop
    size = min(h, w)
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    crop = frame_bgr[start_y:start_y + size, start_x:start_x + size]

    metadata.update({
        "bbox": [int(start_x), int(start_y), int(start_x + size), int(start_y + size)],
        "bbox_padded": [int(start_x), int(start_y), int(start_x + size), int(start_y + size)],
        "crop_method": "center_crop"
    })

    return crop, metadata


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


# ============================================================
# FEATURE EXTRACTION - SESUAI V3
# ============================================================

def aggregate_sequence_stats(seq: np.ndarray) -> np.ndarray:
    """
    Agregasi sequence seperti alur V3:
    mean, std, max, min, median, p10, p90, delta_mean, delta_std, delta_max.

    Output length = 10 * jumlah fitur per frame.
    """
    seq = np.asarray(seq, dtype=np.float32)

    if seq.ndim == 1:
        seq = seq.reshape(1, -1)

    mean = seq.mean(axis=0)
    std = seq.std(axis=0)
    maxv = seq.max(axis=0)
    minv = seq.min(axis=0)
    med = np.median(seq, axis=0)
    p10 = np.percentile(seq, 10, axis=0)
    p90 = np.percentile(seq, 90, axis=0)

    if seq.shape[0] > 1:
        delta = np.abs(np.diff(seq, axis=0))
        d_mean = delta.mean(axis=0)
        d_std = delta.std(axis=0)
        d_max = delta.max(axis=0)
    else:
        d_mean = np.zeros_like(mean)
        d_std = np.zeros_like(std)
        d_max = np.zeros_like(maxv)

    return np.concatenate([
        mean, std, maxv, minv, med, p10, p90,
        d_mean, d_std, d_max
    ]).astype(np.float32)


def extract_basic_visual_features_from_frame(face_bgr: np.ndarray) -> np.ndarray:
    """
    Fitur visual dasar per frame.
    Fitur ini disejajarkan dengan scaler feature_scaler.pkl.
    Default menghasilkan 10 fitur. Jika scaler mengharapkan jumlah lain,
    akan dipad/truncate sebelum transform.
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


def make_lbp_histogram(gray: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    LBP sederhana tanpa dependency tambahan.
    Output histogram dinormalisasi.
    """
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)

    center = gray[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)

    neighbors = [
        gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:],
        gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
        gray[2:, :-2], gray[1:-1, :-2],
    ]

    for i, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << i)

    hist, _ = np.histogram(code.ravel(), bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-6)
    return hist


def blockiness_features(gray: np.ndarray) -> np.ndarray:
    """
    Mengukur indikasi blocking artifact sederhana pada batas blok 8x8.
    """
    gray_f = gray.astype(np.float32)

    vertical_boundaries = gray_f[:, 8::8]
    vertical_prev = gray_f[:, 7::8]
    n_v = min(vertical_boundaries.shape[1], vertical_prev.shape[1])
    if n_v > 0:
        v_diff = np.abs(vertical_boundaries[:, :n_v] - vertical_prev[:, :n_v])
        v_mean = float(np.mean(v_diff))
        v_std = float(np.std(v_diff))
    else:
        v_mean = 0.0
        v_std = 0.0

    horizontal_boundaries = gray_f[8::8, :]
    horizontal_prev = gray_f[7::8, :]
    n_h = min(horizontal_boundaries.shape[0], horizontal_prev.shape[0])
    if n_h > 0:
        h_diff = np.abs(horizontal_boundaries[:n_h, :] - horizontal_prev[:n_h, :])
        h_mean = float(np.mean(h_diff))
        h_std = float(np.std(h_diff))
    else:
        h_mean = 0.0
        h_std = 0.0

    return np.array([
        v_mean,
        v_std,
        h_mean,
        h_std,
        (v_mean + h_mean) / 2.0,
    ], dtype=np.float32)


def dct_frequency_features(gray: np.ndarray) -> np.ndarray:
    """
    Fitur frekuensi DCT sederhana untuk menangkap pola kompresi/frequency artifact.
    """
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dct = np.abs(cv2.dct(small))

    low = dct[:8, :8]
    mid = dct[8:24, 8:24]
    high = dct[24:, 24:]

    def stats(x):
        return [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.max(x)),
            float(np.percentile(x, 90)),
        ]

    return np.array(stats(low) + stats(mid) + stats(high), dtype=np.float32)


def extract_rich_artifact_features_from_frame(face_bgr: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Fitur artifact visual per frame.
    Fitur ini dibuat cukup kaya, lalu dipad/truncate agar cocok dengan
    sisa dimensi yang diharapkan classifier V3.

    Catatan:
    - Artifact tidak di-scaler dengan feature_scaler.pkl.
    - Perubahan antar-frame direpresentasikan oleh aggregate_sequence_stats()
      melalui delta_mean, delta_std, dan delta_max.
    """
    if target_dim <= 0:
        return np.array([], dtype=np.float32)

    resized = cv2.resize(face_bgr, (160, 160), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

    gray_f = gray.astype(np.float32)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 80, 160)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Residual noise sederhana: gray - gaussian blur.
    blur = cv2.GaussianBlur(gray_f, (5, 5), 0)
    residual = gray_f - blur

    features: List[float] = []

    # Statistik gray dasar.
    features.extend([
        float(np.mean(gray_f)),
        float(np.std(gray_f)),
        float(np.min(gray_f)),
        float(np.max(gray_f)),
        float(np.median(gray_f)),
        float(np.percentile(gray_f, 10)),
        float(np.percentile(gray_f, 90)),
    ])

    # Ketajaman/edge.
    features.extend([
        float(np.var(lap)),
        float(np.mean(np.abs(lap))),
        float(np.std(lap)),
        float(np.mean(edges > 0)),
        float(np.mean(edges)),
        float(np.std(edges)),
        float(np.mean(grad_mag)),
        float(np.std(grad_mag)),
        float(np.percentile(grad_mag, 90)),
    ])

    # Residual noise.
    features.extend([
        float(np.mean(residual)),
        float(np.std(residual)),
        float(np.mean(np.abs(residual))),
        float(np.percentile(np.abs(residual), 90)),
    ])

    # Statistik warna RGB, HSV, YCrCb, LAB.
    for image in (rgb, hsv, ycrcb, lab):
        for c in range(3):
            channel = image[:, :, c].astype(np.float32)
            features.extend([
                float(np.mean(channel)),
                float(np.std(channel)),
                float(np.percentile(channel, 10)),
                float(np.percentile(channel, 90)),
            ])

    # Blockiness.
    features.extend(blockiness_features(gray).tolist())

    # DCT.
    features.extend(dct_frequency_features(gray).tolist())

    # LBP histogram.
    features.extend(make_lbp_histogram(gray, bins=16).tolist())

    return pad_or_truncate(np.array(features, dtype=np.float32), target_dim)


def build_v3_feature_vector(embeddings: np.ndarray, face_crops_bgr: List[np.ndarray]) -> np.ndarray:
    """
    Membangun fitur video final untuk classifier V3.

    Susunan fitur yang dipakai:
    1. Agregasi embedding Xception 2048-D  -> 2048 * 10 = 20480 fitur
    2. Agregasi fitur visual dasar         -> scaler.n_features_in_ * 10
    3. Agregasi fitur artifact visual      -> sisa dimensi classifier

    Ini lebih sesuai dengan alur training V3 dibanding hanya padding/truncate
    embedding sampai cocok.
    """
    expected_total_features = int(getattr(CLASSIFIER_MODEL, "n_features_in_", 21250))

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        embeddings = embeddings.reshape((embeddings.shape[0], -1))

    xception_agg = aggregate_sequence_stats(embeddings)

    scaler_features = int(getattr(SCALER, "n_features_in_", 10))

    basic_seq = []
    for crop in face_crops_bgr:
        basic = extract_basic_visual_features_from_frame(crop)
        basic = pad_or_truncate(basic, scaler_features)
        basic_seq.append(basic)

    basic_seq = np.asarray(basic_seq, dtype=np.float32)

    # feature_scaler.pkl dipakai pada fitur dasar per frame, bukan pada agregasi video.
    try:
        basic_seq_scaled = SCALER.transform(basic_seq).astype(np.float32)
    except Exception:
        # Fallback agar API tetap jalan jika scaler tidak cocok.
        basic_seq_scaled = basic_seq.astype(np.float32)

    basic_agg = aggregate_sequence_stats(basic_seq_scaled)

    remaining = expected_total_features - xception_agg.shape[0] - basic_agg.shape[0]

    if remaining > 0:
        if remaining % 10 == 0:
            artifact_dim = remaining // 10
        else:
            artifact_dim = max(1, int(np.ceil(remaining / 10)))

        artifact_seq = []
        for crop in face_crops_bgr:
            artifact = extract_rich_artifact_features_from_frame(crop, artifact_dim)
            artifact_seq.append(artifact)

        artifact_seq = np.asarray(artifact_seq, dtype=np.float32)
        artifact_agg = aggregate_sequence_stats(artifact_seq)
        artifact_agg = pad_or_truncate(artifact_agg, remaining)
    else:
        artifact_agg = np.array([], dtype=np.float32)

    final_features = np.concatenate([xception_agg, basic_agg, artifact_agg], axis=0).astype(np.float32)
    final_features = pad_or_truncate(final_features, expected_total_features)

    return final_features.reshape(1, -1)


def extract_features_from_video(video_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Proses utama ekstraksi fitur:
    1. Ambil 24 frame dari video
    2. Deteksi/crop wajah dengan YOLO
    3. Encode crop wajah dengan Xception encoder
    4. Ekstraksi fitur visual dasar + artifact
    5. Gabungkan fitur sesuai kebutuhan classifier V3
    """
    seq_len = get_seq_len()
    frames = read_video_frames(video_path, seq_len)

    face_crops_bgr: List[np.ndarray] = []
    xception_inputs: List[np.ndarray] = []
    frame_infos: List[Dict[str, Any]] = []

    face_detected_count = 0
    center_crop_count = 0

    for frame_time, frame_bgr, repeated_frame in frames:
        face_crop, meta = crop_face_with_yolo(frame_bgr)

        if meta.get("face_detected"):
            face_detected_count += 1
        else:
            center_crop_count += 1

        face_crops_bgr.append(face_crop)

        x_input = prepare_face_for_xception(face_crop)
        xception_inputs.append(x_input)

        frame_infos.append({
            "frame_time": frame_time,
            "repeated_frame": bool(repeated_frame),
            "face_detected": bool(meta.get("face_detected")),
            "face_confidence": meta.get("face_confidence"),
            "bbox": meta.get("bbox"),
            "bbox_padded": meta.get("bbox_padded"),
            "crop_method": meta.get("crop_method"),
        })

    xception_batch = np.asarray(xception_inputs, dtype=np.float32)

    embeddings = XCEPTION_ENCODER.predict(xception_batch, verbose=0)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    features = build_v3_feature_vector(embeddings, face_crops_bgr)

    feature_debug = {
        "frames_requested": seq_len,
        "frames_used": len(frames),
        "face_detected_count": face_detected_count,
        "center_crop_count": center_crop_count,
        "xception_embedding_shape": list(embeddings.shape),
        "feature_vector_shape": list(features.shape),
        "classifier_expected_features": int(getattr(CLASSIFIER_MODEL, "n_features_in_", 0)),
        "scaler_features": int(getattr(SCALER, "n_features_in_", 0)),
    }

    return features, frame_infos, feature_debug


# ============================================================
# PREDICTION
# ============================================================

def is_v21_payload() -> bool:
    return (
        isinstance(CLASSIFIER_PAYLOAD, dict)
        and "base_model" in CLASSIFIER_PAYLOAD
        and V21_X_REF_NORM is not None
        and V21_Y_REF is not None
        and isinstance(V21_CONFIG, dict)
        and len(V21_CONFIG) > 0
    )


def l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norm + 1e-8)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def get_model_fake_score(model, features: np.ndarray) -> float:
    """Ambil probabilitas class 1/FAKE dari model sklearn."""
    features = np.asarray(features, dtype=np.float32)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        classes = list(model.classes_)

        if 1 in classes:
            return float(proba[classes.index(1)])

        return float(proba[-1])

    pred = int(model.predict(features)[0])
    return 1.0 if pred == 1 else 0.0


def apply_flip_score(score_fake: float, use_flip: bool) -> float:
    return float(1.0 - score_fake) if use_flip else float(score_fake)


def knn_prob_fake_single(features: np.ndarray, k: int = 15, temp: float = 0.05, power: float = 1.0) -> float:
    """Local similarity correction milik V21."""
    if V21_X_REF_NORM is None or V21_Y_REF is None:
        return 0.5

    x_norm = l2_normalize(features)
    sims = x_norm @ V21_X_REF_NORM.T

    k = int(min(max(1, k), V21_X_REF_NORM.shape[0]))
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]

    row_ids = np.arange(sims.shape[0])[:, None]
    top_sims = sims[row_ids, idx]
    top_labels = V21_Y_REF[idx]

    z = (top_sims - top_sims.max(axis=1, keepdims=True)) / float(temp)
    weights = np.exp(z)

    if float(power) != 1.0:
        weights = weights ** float(power)

    prob = (weights * top_labels).sum(axis=1) / (weights.sum(axis=1) + 1e-8)
    return float(prob[0])


def blend_prob(base_prob: float, local_prob: float, alpha: float = 0.75, mode: str = "linear") -> float:
    base_prob = float(base_prob)
    local_prob = float(local_prob)
    alpha = float(alpha)

    if mode == "linear":
        return float((alpha * base_prob) + ((1.0 - alpha) * local_prob))

    if mode == "logit":
        return float(sigmoid((alpha * logit(base_prob)) + ((1.0 - alpha) * logit(local_prob))))

    if mode == "borderline":
        center = 0.49
        dist = abs(base_prob - center)
        gate = np.clip(1.0 - (dist / 0.12), 0.0, 1.0)
        mixed = (alpha * base_prob) + ((1.0 - alpha) * local_prob)
        return float(((1.0 - gate) * base_prob) + (gate * mixed))

    return float((alpha * base_prob) + ((1.0 - alpha) * local_prob))


def predict_with_v21(features: np.ndarray) -> Dict[str, Any]:
    """Prediksi memakai V21 = base model + local similarity correction."""
    threshold = get_threshold()
    features = np.asarray(features, dtype=np.float32).reshape(1, -1)

    expected_features = int(getattr(CLASSIFIER_MODEL, "n_features_in_", features.shape[1]))
    if features.shape[1] != expected_features:
        features = pad_or_truncate(features.ravel(), expected_features).reshape(1, -1)

    raw_fake = get_model_fake_score(CLASSIFIER_MODEL, features)
    base_score_fake = apply_flip_score(raw_fake, V21_BASE_USE_FLIP)

    local_score_fake = knn_prob_fake_single(
        features,
        k=int(V21_CONFIG.get("k", 15)),
        temp=float(V21_CONFIG.get("temp", 0.05)),
        power=float(V21_CONFIG.get("power", 1.0))
    )

    fake_score = blend_prob(
        base_score_fake,
        local_score_fake,
        alpha=float(V21_CONFIG.get("alpha", 0.75)),
        mode=str(V21_CONFIG.get("mode", "linear"))
    )

    real_score = float(1.0 - fake_score)

    if fake_score >= threshold:
        prediction = "DEEPFAKE"
        label = "DEEPFAKE"
        confidence = fake_score
    else:
        prediction = "REAL"
        confidence = real_score
        # Supaya fake realistis yang dekat batas tidak langsung dianggap aman.
        if fake_score >= V21_SUSPICIOUS_MIN:
            label = "MENCURIGAKAN"
        else:
            label = "REAL"

    margin = abs(fake_score - threshold)

    if label == "MENCURIGAKAN":
        confidence_note = "Mencurigakan / perlu review manual"
        decision_explanation = (
            "fake_score belum melewati threshold DEEPFAKE, tetapi berada di area mencurigakan. "
            "Sistem menyarankan review manual."
        )
    elif margin < 0.05:
        confidence_note = "Kurang yakin / dekat threshold"
        decision_explanation = "Keputusan dekat dengan threshold model."
    elif margin < 0.12:
        confidence_note = "Cukup yakin"
        decision_explanation = "Keputusan cukup jauh dari threshold model."
    else:
        confidence_note = "Yakin"
        decision_explanation = "Keputusan jauh dari threshold model."

    return {
        "prediction": prediction,
        "label": label,
        "status": label,
        "confidence": round(float(confidence), 6),
        "real_score": round(float(real_score), 6),
        "fake_score": round(float(fake_score), 6),
        "base_score_fake": round(float(base_score_fake), 6),
        "local_score_fake": round(float(local_score_fake), 6),
        "threshold": round(float(threshold), 6),
        "margin": round(float(margin), 6),
        "confidence_note": confidence_note,
        "decision_rule": "DEEPFAKE jika fake_score >= threshold; MENCURIGAKAN jika 0.40 <= fake_score < threshold; REAL jika fake_score < 0.40",
        "decision_explanation": decision_explanation,
        "model_version": MODEL_VERSION,
    }


def predict_with_classifier(features: np.ndarray) -> Dict[str, Any]:
    """
    Prediksi REAL/DEEPFAKE.
    Otomatis memakai alur V21 jika payload model adalah V21.
    Kalau bukan V21, fallback ke alur V3 lama.
    """
    if is_v21_payload():
        return predict_with_v21(features)

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
        label = "DEEPFAKE"
        confidence = fake_score
    else:
        prediction = "REAL"
        label = "REAL"
        confidence = real_score

    margin = abs(fake_score - threshold)

    if margin < 0.05:
        confidence_note = "Kurang yakin / dekat threshold"
    elif margin < 0.12:
        confidence_note = "Cukup yakin"
    else:
        confidence_note = "Yakin"

    if prediction == "DEEPFAKE" and real_score > fake_score:
        decision_explanation = (
            "Label DEEPFAKE karena fake_score melewati threshold, "
            "meskipun real_score lebih besar dari fake_score."
        )
    elif prediction == "DEEPFAKE":
        decision_explanation = "Label DEEPFAKE karena fake_score >= threshold."
    else:
        decision_explanation = "Label REAL karena fake_score < threshold."

    return {
        "prediction": prediction,
        "label": label,
        "status": label,
        "confidence": round(float(confidence), 6),
        "real_score": round(float(real_score), 6),
        "fake_score": round(float(fake_score), 6),
        "threshold": round(float(threshold), 6),
        "margin": round(float(margin), 6),
        "confidence_note": confidence_note,
        "decision_rule": "DEEPFAKE jika fake_score >= threshold, REAL jika fake_score < threshold",
        "decision_explanation": decision_explanation,
        "model_version": MODEL_VERSION,
    }

# ============================================================
# ROUTES
# ============================================================

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
        "threshold": get_threshold() if MODEL_READY else None,
        "model_version": MODEL_VERSION,
        "v21_correction_config": V21_CONFIG if is_v21_payload() else None,
        "v21_reference_shape": list(V21_X_REF_NORM.shape) if V21_X_REF_NORM is not None else None,
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

        features, frame_infos, feature_debug = extract_features_from_video(save_path)

        # ====================================================
        # GUARD PENTING:
        # Model V3 dilatih untuk video yang memiliki wajah.
        # Jika YOLO tidak menemukan wajah atau wajah terlalu sedikit,
        # jangan paksa klasifikasi sebagai REAL/DEEPFAKE.
        # ====================================================
        min_face_frames = get_min_face_frames()
        face_detected_count = int(feature_debug.get("face_detected_count", 0))

        if face_detected_count < min_face_frames:
            duration_seconds = round(time.time() - start_time, 2)

            frames_response = []
            for frame in frame_infos:
                frames_response.append({
                    "frame_time": frame["frame_time"],
                    "status": "wajah tidak terdeteksi" if not frame["face_detected"] else "wajah terdeteksi",
                    "face_detected": frame["face_detected"],
                    "face_confidence": frame["face_confidence"],
                    "crop_method": frame["crop_method"],
                    "repeated_frame": frame["repeated_frame"],
                    "bbox": frame["bbox"],
                    "note": "Video tidak diklasifikasikan karena jumlah frame wajah tidak mencukupi."
                })

            return jsonify({
                "success": True,
                "prediction": "NO_FACE",
                "label": "NO_FACE",
                "confidence": 0.0,
                "real_score": None,
                "fake_score": None,
                "threshold": get_threshold(),
                "margin": None,
                "confidence_note": "Wajah tidak terdeteksi / frame wajah tidak mencukupi",
                "decision_rule": "Klasifikasi hanya dilakukan jika wajah terdeteksi minimal pada beberapa frame.",
                "decision_explanation": (
                    f"Video tidak diklasifikasikan karena hanya {face_detected_count} "
                    f"frame wajah terdeteksi dari minimal {min_face_frames} frame yang dibutuhkan."
                ),
                "duration_seconds": duration_seconds,
                "message": "Wajah tidak terdeteksi atau tidak cukup jelas. Upload video wajah untuk dianalisis.",
                "frames_used": len(frame_infos),
                "face_detected_count": face_detected_count,
                "min_face_frames": min_face_frames,
                "feature_debug": feature_debug,
                "frames": frames_response
            })

        result = predict_with_classifier(features)

        duration_seconds = round(time.time() - start_time, 2)

        # Detail frame:
        # Model V3 menghasilkan score pada level video, bukan score frame asli.
        # Agar website tidak menyesatkan, frame tetap dikirim dengan note.
        frames_response = []
        for frame in frame_infos:
            frames_response.append({
                "frame_time": frame["frame_time"],
                "status": "frame digunakan",
                "face_detected": frame["face_detected"],
                "face_confidence": frame["face_confidence"],
                "crop_method": frame["crop_method"],
                "repeated_frame": frame["repeated_frame"],
                "bbox": frame["bbox"],
                "note": "Score prediksi dihitung pada level video, bukan per-frame."
            })

        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "label": result.get("label", result["prediction"]),
            "status": result.get("status", result.get("label", result["prediction"])),
            "confidence": result["confidence"],
            "real_score": result["real_score"],
            "fake_score": result["fake_score"],
            "base_score_fake": result.get("base_score_fake"),
            "local_score_fake": result.get("local_score_fake"),
            "threshold": result["threshold"],
            "margin": result["margin"],
            "confidence_note": result["confidence_note"],
            "decision_rule": result["decision_rule"],
            "decision_explanation": result["decision_explanation"],
            "model_version": result.get("model_version", MODEL_VERSION),
            "duration_seconds": duration_seconds,
            "message": "Prediksi berhasil",
            "frames_used": len(frame_infos),
            "feature_debug": feature_debug,
            "frames": frames_response
        })

    except Exception as error:
        traceback.print_exc()

        return jsonify({
            "success": False,
            "message": f"Video tidak dapat diproses: {str(error)}"
        }), 500


# Load otomatis saat module diimport oleh server production.
load_all_models()


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    app.run(host=host, port=port, debug=debug)
