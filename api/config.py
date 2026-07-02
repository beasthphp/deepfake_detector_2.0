from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from face_pipeline.cropper import SELECTED_CROP_STRATEGY
from model_runtime.metadata import ScoreThresholds
from model_runtime.registry import DEFAULT_ARTIFACT_DIR, DEFAULT_REGISTRY_PATH


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class APIConfig:
    active_model_id: str = "legacy-cnn-v1"
    model_registry_path: Path = DEFAULT_REGISTRY_PATH
    model_artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    model_path: Path | None = None
    model_id: str = "legacy-cnn-v1"
    model_version: str = "1"
    detector_id: str = "opencv_haar_frontalface_default"
    detector_version: str = "1"
    threshold_id: str = "score-thresholds"
    threshold_version: str = "1"
    crop_strategy_version: str = "phase2.1"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    min_image_width: int = 32
    min_image_height: int = 32
    max_image_width: int = 10_000
    max_image_height: int = 10_000
    max_faces_per_image: int = 20
    request_concurrency_limit: int = 2
    request_timeout_seconds: float = 30.0
    save_debug_outputs: bool = False
    allowed_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost",
            "http://127.0.0.1",
        ]
    )
    thresholds: ScoreThresholds = field(default_factory=ScoreThresholds)
    selected_crop_strategy: str = SELECTED_CROP_STRATEGY

    @classmethod
    def from_env(cls) -> "APIConfig":
        active_model_id = os.getenv("ACTIVE_MODEL_ID") or os.getenv("DEEPFAKE_MODEL_ID") or "legacy-cnn-v1"
        return cls(
            active_model_id=active_model_id,
            model_registry_path=Path(os.getenv("MODEL_REGISTRY_PATH", str(DEFAULT_REGISTRY_PATH))),
            model_artifact_dir=Path(os.getenv("MODEL_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))),
            model_path=_env_optional_path("DEEPFAKE_MODEL_PATH"),
            model_id=active_model_id,
            model_version=os.getenv("DEEPFAKE_MODEL_VERSION", "1"),
            detector_id=os.getenv("DEEPFAKE_DETECTOR_ID", "opencv_haar_frontalface_default"),
            detector_version=os.getenv("DEEPFAKE_DETECTOR_VERSION", "1"),
            threshold_id=os.getenv("DEEPFAKE_THRESHOLD_ID", "score-thresholds"),
            threshold_version=os.getenv("DEEPFAKE_THRESHOLD_VERSION", "1"),
            crop_strategy_version=os.getenv("DEEPFAKE_CROP_STRATEGY_VERSION", "phase2.1"),
            max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
            max_image_pixels=_env_int("MAX_IMAGE_PIXELS", 25_000_000),
            min_image_width=_env_int("MIN_IMAGE_WIDTH", 32),
            min_image_height=_env_int("MIN_IMAGE_HEIGHT", 32),
            max_image_width=_env_int("MAX_IMAGE_WIDTH", 10_000),
            max_image_height=_env_int("MAX_IMAGE_HEIGHT", 10_000),
            max_faces_per_image=_env_int("MAX_FACES_PER_IMAGE", 20),
            request_concurrency_limit=max(1, _env_int("REQUEST_CONCURRENCY_LIMIT", 2)),
            request_timeout_seconds=max(0.1, _env_float("REQUEST_TIMEOUT_SECONDS", 30.0)),
            save_debug_outputs=_env_bool("SAVE_DEBUG_OUTPUTS", False),
            allowed_origins=_env_list(
                "ALLOWED_ORIGINS",
                ["http://localhost", "http://127.0.0.1"],
            ),
        )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_optional_path(name: str) -> Path | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return Path(raw)
