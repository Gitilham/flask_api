from typing import Any

from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    success: bool
    prediction: str
    label: str
    confidence: float
    real_score: float | None
    fake_score: float | None
    threshold: float
    margin: float | None
    message: str
    frames_used: int
    feature_debug: dict[str, Any]
    frames: list[dict[str, Any]]

