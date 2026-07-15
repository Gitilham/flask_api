from typing import Any

from pydantic import BaseModel, ConfigDict


class FlexibleResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    success: bool | None = None
    status: str | None = None
    message: str | None = None


class ModelStatusResponse(BaseModel):
    api_status: str
    model_loaded: bool
    model_version: str | None
    threshold: float | None
    label_map: dict[str, str]
    local_similarity_mode: str | None
    file_availability: dict[str, bool]
    load_duration_seconds: float | None
    runtime_device: str
    library_versions: dict[str, str | None]
    ready: bool
    error: str | None

