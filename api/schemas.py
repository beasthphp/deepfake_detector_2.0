from __future__ import annotations

from pydantic import BaseModel, Field


class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ImageInfo(BaseModel):
    width: int
    height: int
    format: str


class VersionedComponent(BaseModel):
    id: str = "unknown"
    version: str = "unknown"


class ThresholdMetadata(VersionedComponent):
    likely_fake_below: float | None = None
    likely_real_above: float | None = None


class FacePrediction(BaseModel):
    face_index: int
    bounding_box: Box
    crop_box: Box
    face_detection_score: float | None = None
    crop_strategy: str
    preserved_original: bool
    real_score: float = Field(ge=0.0, le=1.0)
    fake_score: float = Field(ge=0.0, le=1.0)
    label: str


class TimingInfo(BaseModel):
    decode: float = 0.0
    face_detection: float = 0.0
    crop_preprocessing: float = 0.0
    classification: float = 0.0
    serialization: float = 0.0
    total: float = 0.0


class PredictionResponse(BaseModel):
    request_id: str
    status: str
    model: VersionedComponent = Field(default_factory=VersionedComponent)
    detector: VersionedComponent = Field(default_factory=VersionedComponent)
    crop_strategy: VersionedComponent = Field(default_factory=VersionedComponent)
    thresholds: ThresholdMetadata = Field(default_factory=ThresholdMetadata)
    image: ImageInfo
    faces_detected: int
    faces: list[FacePrediction]
    timing_ms: TimingInfo
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detector_loaded: bool
    device: str
    model_id: str | None = None
    model_version: str
    detector_id: str | None = None
    detector_version: str | None = None
    crop_strategy_id: str | None = None
    crop_strategy_version: str | None = None
    threshold_id: str | None = None
    threshold_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_id: str | None = None
    model_version: str
    input_shape: list[int | None]
    output_shape: list[int | None]
    output_meaning: str
    class_mapping: dict[str, int]
    selected_face_detector: str
    selected_face_detector_version: str | None = None
    selected_crop_strategy: str
    selected_crop_strategy_version: str | None = None
    threshold_version: str | None = None
    provisional_thresholds: dict[str, str]
    known_limitations: list[str]


class ErrorResponse(BaseModel):
    request_id: str | None = None
    error: dict[str, str]
