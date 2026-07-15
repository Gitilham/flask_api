from fastapi import APIRouter, Request

from app.services import v21_pipeline as pipeline

router = APIRouter(tags=["models"])


@router.get("/models/status")
async def models_status(request: Request):
    return request.app.state.model_registry.public_status()


@router.get("/model-info", include_in_schema=False)
async def legacy_model_info(request: Request):
    registry = request.app.state.model_registry
    info = {
        "success": registry.ready, "model_ready": registry.ready, "model_error": registry.error,
        "config": pipeline.CONFIG, "threshold": pipeline.get_threshold() if registry.ready else None,
        "model_version": pipeline.MODEL_VERSION,
        "v21_correction_config": pipeline.V21_CONFIG if pipeline.is_v21_payload() else None,
        "v21_reference_shape": list(pipeline.V21_X_REF_NORM.shape) if pipeline.V21_X_REF_NORM is not None else None,
        "paths": {name + "_available": available for name, available in registry.file_availability.items()},
    }
    if pipeline.CLASSIFIER_MODEL is not None:
        info["classifier_type"] = type(pipeline.CLASSIFIER_MODEL).__name__
        info["classifier_features"] = int(getattr(pipeline.CLASSIFIER_MODEL, "n_features_in_", 0))
        info["classifier_classes"] = [int(x) for x in getattr(pipeline.CLASSIFIER_MODEL, "classes_", [])]
    if pipeline.SCALER is not None:
        info["scaler_features"] = int(getattr(pipeline.SCALER, "n_features_in_", 0))
    return info

