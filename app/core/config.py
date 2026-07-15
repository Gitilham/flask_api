import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "Deepfake Detection API"
    app_version: str = "21.0.0"
    host: str = "0.0.0.0"
    port: int = 5000
    log_level: str = "INFO"
    cors_allowed_origins: str = "http://localhost:8080"
    upload_folder: Path = Path("uploads")
    temp_folder: Path = Path("temp")
    max_content_length_mb: int = 300
    upload_chunk_bytes: int = 1024 * 1024
    preserve_uploads: bool = False
    inference_concurrency: int = 1
    inference_queue_timeout: float = 900.0
    max_upload_queue: int = 5
    config_path: Path = Path("models/config.json")
    model_path: Path = Path("models/best_v21_manual_audit_local_similarity.pkl")
    scaler_path: Path = Path("models/feature_scaler.pkl")
    yolo_path: Path = Path("models/face_yolov8n.pt")
    xception_encoder_path: Path = Path("models/xception_frame_encoder_safe.keras")
    model_metadata_path: Path = Path("models/model_metadata.json")
    tensorflow_threads: int = 1
    torch_threads: int = 1

    @property
    def cors_origins(self) -> list[str]:
        return [value.strip() for value in self.cors_allowed_origins.split(",") if value.strip()]

    @property
    def required_model_files(self) -> dict[str, Path]:
        return {
            "config": self.config_path,
            "classifier": self.model_path,
            "scaler": self.scaler_path,
            "yolo": self.yolo_path,
            "xception": self.xception_encoder_path,
        }


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    os.environ.setdefault("OMP_NUM_THREADS", str(settings.torch_threads))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(settings.tensorflow_threads))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(settings.tensorflow_threads))
    return settings

