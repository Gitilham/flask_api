import os
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings
from app.exceptions.handlers import ApiError

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
ALLOWED_MIME_TYPES = {
    "video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska",
    "application/octet-stream",  # Beberapa client lama tidak mengirim MIME video spesifik.
}


async def stream_to_temporary(upload: UploadFile, settings: Settings) -> tuple[Path, int, str]:
    original_name = Path(upload.filename or "").name
    if not original_name:
        raise ApiError("Nama file video kosong.", "EMPTY_FILENAME", 400)
    suffix = Path(original_name).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_EXTENSIONS:
        raise ApiError("Format video tidak didukung. Gunakan mp4, avi, mov, atau mkv.", "INVALID_EXTENSION", 400)
    if upload.content_type and upload.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise ApiError("MIME type video tidak didukung.", "INVALID_MIME", 400)

    settings.temp_folder.mkdir(parents=True, exist_ok=True)
    temporary_path = settings.temp_folder / f"{uuid.uuid4().hex}.{suffix}"
    maximum = settings.max_content_length_mb * 1024 * 1024
    size = 0
    try:
        with temporary_path.open("xb") as target:
            while chunk := await upload.read(settings.upload_chunk_bytes):
                size += len(chunk)
                if size > maximum:
                    raise ApiError(
                        f"Ukuran video melebihi batas {settings.max_content_length_mb} MB.",
                        "FILE_TOO_LARGE", 413,
                    )
                target.write(chunk)
        if size == 0:
            raise ApiError("File video kosong.", "EMPTY_FILE", 400)
        return temporary_path, size, original_name
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def cleanup_file(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)

