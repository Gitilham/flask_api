from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["status"])


@router.get("/")
async def index(request: Request):
    registry = request.app.state.model_registry
    return {
        "success": True,
        "message": "Flask API Deepfake Detection aktif",
        "endpoint": "/predict-video",
        "model_ready": registry.ready,
        "model_error": registry.error,
        "model_name": registry.settings.app_name,
    }


@router.get("/health")
async def health():
    return {"success": True, "status": "healthy", "message": "API berjalan normal", "model_error": None}


@router.get("/ready")
async def ready(request: Request):
    registry = request.app.state.model_registry
    content = {
        "success": registry.ready,
        "status": "ready" if registry.ready else "not_ready",
        "message": "Model siap" if registry.ready else "Model belum siap",
        "model_error": registry.error,
    }
    return JSONResponse(status_code=200 if registry.ready else 503, content=content)
