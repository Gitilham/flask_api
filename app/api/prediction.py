import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile

from app.exceptions.handlers import ApiError
from app.services.file_service import cleanup_file, stream_to_temporary
from app.services.prediction_service import predict_video

router = APIRouter(tags=["prediction"])
logger = logging.getLogger(__name__)


@router.post("/predict-video")
@router.post("/api/v1/predict-video", include_in_schema=False)
async def predict(request: Request, video: UploadFile | None = File(default=None)):
    if video is None:
        raise ApiError("File video tidak ditemukan. Gunakan field bernama video.", "VIDEO_REQUIRED", 400)
    registry = request.app.state.model_registry
    if not registry.ready:
        raise ApiError("Model belum siap atau gagal dimuat.", "MODEL_NOT_READY", 500)

    settings = registry.settings
    temporary_path: Path | None = None
    queued = False
    started = time.perf_counter()
    try:
        temporary_path, size, safe_name = await stream_to_temporary(video, settings)
        async with request.app.state.queue_lock:
            if request.app.state.queued_requests >= settings.max_upload_queue:
                raise ApiError("Antrean inferensi penuh. Coba kembali nanti.", "QUEUE_FULL", 429)
            request.app.state.queued_requests += 1
            queued = True
        try:
            await asyncio.wait_for(
                request.app.state.inference_semaphore.acquire(),
                timeout=settings.inference_queue_timeout,
            )
        except TimeoutError as exc:
            raise ApiError("Waktu tunggu antrean inferensi habis.", "QUEUE_TIMEOUT", 503) from exc
        finally:
            if queued:
                async with request.app.state.queue_lock:
                    request.app.state.queued_requests -= 1
                queued = False
        try:
            result = await asyncio.to_thread(predict_video, temporary_path)
        finally:
            request.app.state.inference_semaphore.release()
        logger.info("prediction_completed", extra={"context": {
            "request_id": request.state.request_id, "endpoint": "/predict-video",
            "filename": safe_name, "file_size": size, "processing_seconds": round(time.perf_counter() - started, 6),
            "frames": result.get("frames_used"), "faces": result.get("feature_debug", {}).get("face_detected_count"),
            "result": result.get("label"), "confidence": result.get("confidence"), "status": "success",
        }})
        return result
    finally:
        if queued:
            async with request.app.state.queue_lock:
                request.app.state.queued_requests -= 1
        if not settings.preserve_uploads:
            cleanup_file(temporary_path)

