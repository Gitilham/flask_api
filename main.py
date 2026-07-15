import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.models import router as models_router
from app.api.prediction import router as prediction_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.exceptions.handlers import register_exception_handlers

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

# Imported only after thread-related environment settings are applied.
from app.core.model_registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry(settings)
    app.state.model_registry = registry
    app.state.inference_semaphore = asyncio.Semaphore(max(1, settings.inference_concurrency))
    app.state.queue_lock = asyncio.Lock()
    app.state.queued_requests = 0
    settings.temp_folder.mkdir(parents=True, exist_ok=True)
    if settings.preserve_uploads:
        settings.upload_folder.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.to_thread(registry.load)
        logger.info("model_registry_loaded", extra={"context": registry.public_status()})
    except Exception:
        logger.exception("model_registry_load_failed")
    yield
    registry.unload()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API deteksi video Deepfake V21 yang kompatibel dengan endpoint Flask lama.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(health_router)
app.include_router(models_router)
app.include_router(prediction_router)
register_exception_handlers(app)
