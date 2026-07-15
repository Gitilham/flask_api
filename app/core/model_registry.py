import importlib.metadata
import json
import logging
import time
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services import v21_pipeline as pipeline

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded = False
        self.error: str | None = None
        self.load_duration_seconds: float | None = None
        self.file_availability: dict[str, bool] = {}

    def load(self) -> None:
        if self.loaded:
            return
        started = time.perf_counter()
        self.file_availability = {
            name: Path(path).is_file() for name, path in self.settings.required_model_files.items()
        }
        missing = [name for name, available in self.file_availability.items() if not available]
        if missing:
            self.error = "Komponen model tidak tersedia: " + ", ".join(missing)
            self.load_duration_seconds = round(time.perf_counter() - started, 6)
            raise RuntimeError(self.error)

        pipeline.configure_paths(self.settings)
        pipeline.load_all_models()
        self.load_duration_seconds = round(time.perf_counter() - started, 6)
        self.loaded = bool(pipeline.MODEL_READY)
        self.error = pipeline.MODEL_ERROR
        if not self.loaded:
            raise RuntimeError(self.error or "Model V21 gagal dimuat")
        try:
            self._validate_contract()
        except Exception as exc:
            self.loaded = False
            self.error = str(exc)
            raise

    def _validate_contract(self) -> None:
        metadata = json.loads(self.settings.model_metadata_path.read_text(encoding="utf-8"))
        expected_shape = int(metadata.get("feature_shape", 21250))
        actual_shape = int(getattr(pipeline.CLASSIFIER_MODEL, "n_features_in_", 0))
        if actual_shape != expected_shape:
            raise RuntimeError(f"Feature shape model {actual_shape}, diharapkan {expected_shape}")
        if pipeline.get_threshold() != 0.5:
            raise RuntimeError("Threshold V21 harus tetap 0.5")
        if metadata.get("label_map") != {"0": "REAL", "1": "FAKE"}:
            raise RuntimeError("Label map V21 tidak sesuai kontrak REAL/FAKE")
        expected_correction = metadata.get("correction_config", {})
        if pipeline.V21_CONFIG != expected_correction:
            raise RuntimeError("Konfigurasi local similarity tidak sesuai metadata V21")

    def unload(self) -> None:
        # Lepaskan referensi tanpa clear_session per request.
        pipeline.CLASSIFIER_PAYLOAD = None
        pipeline.CLASSIFIER_MODEL = None
        pipeline.SCALER = None
        pipeline.YOLO_MODEL = None
        pipeline.XCEPTION_ENCODER = None
        pipeline.MODEL_READY = False
        self.loaded = False

    @property
    def ready(self) -> bool:
        return self.loaded and bool(pipeline.MODEL_READY)

    def public_status(self) -> dict[str, Any]:
        versions = {}
        for package in ("fastapi", "numpy", "opencv-python-headless", "tensorflow", "torch", "ultralytics"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return {
            "api_status": "ready" if self.ready else "not_ready",
            "model_loaded": self.ready,
            "model_version": pipeline.MODEL_VERSION if self.ready else None,
            "threshold": pipeline.get_threshold() if self.ready else None,
            "label_map": {"0": "REAL", "1": "FAKE"},
            "local_similarity_mode": pipeline.V21_CONFIG.get("mode") if self.ready else None,
            "file_availability": self.file_availability,
            "load_duration_seconds": self.load_duration_seconds,
            "runtime_device": "CUDA" if self.ready and pipeline.torch.cuda.is_available() else "CPU",
            "library_versions": versions,
            "ready": self.ready,
            "error": self.error,
        }
