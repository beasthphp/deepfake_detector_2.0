from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from model_runtime.metadata import ScoreThresholds
from model_runtime.providers.tensorflow_binary_face import (
    TensorFlowBinaryFaceProvider,
    ensure_real_hdf5,
    extract_scores,
    extract_single_score,
    label_from_real_score,
    preprocess_binary_face_image,
)
from model_runtime.registry import DEFAULT_ARTIFACT_DIR, DEFAULT_REGISTRY_PATH, create_provider


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "deepfake_detector_93acc.h5"
DEFAULT_MODEL_REGISTRY_PATH = DEFAULT_REGISTRY_PATH
IMAGE_SIZE = (256, 256)


class DeepfakeClassifier:
    """Stable classifier wrapper over a replaceable model provider."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        thresholds: ScoreThresholds | None = None,
        model: Any | None = None,
        provider: Any | None = None,
        active_model_id: str | None = None,
        registry_path: str | Path | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.thresholds = thresholds or ScoreThresholds()
        self.model_path = Path(model_path) if model_path is not None else None
        self.active_model_id = active_model_id
        self.registry_path = Path(registry_path) if registry_path is not None else None
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        if provider is not None:
            self._provider = provider
        elif model is not None:
            self._provider = TensorFlowBinaryFaceProvider.from_loaded_model(model, thresholds=self.thresholds)
        else:
            self._provider = None

    @property
    def provider(self) -> Any:
        if self._provider is None:
            self._provider = create_provider(
                active_model_id=self.active_model_id,
                registry_path=self.registry_path,
                artifact_dir=self.artifact_dir,
                explicit_model_path=self.model_path,
            )
        return self._provider

    @property
    def model(self) -> Any:
        return getattr(self.provider, "model")

    @property
    def metadata(self):
        return self.provider.metadata

    def predict_image(self, image: Image.Image | np.ndarray) -> dict[str, float | str]:
        return self.predict_images([image])[0]

    def predict_images(self, images: list[Image.Image | np.ndarray]) -> list[dict[str, float | str]]:
        return self.provider.predict_batch(images)


def preprocess_for_model(image: Image.Image | np.ndarray) -> np.ndarray:
    return preprocess_binary_face_image(image, (None, 256, 256, 3))


def _extract_single_score(prediction: Any) -> float:
    return extract_single_score(prediction)


def _extract_scores(prediction: Any, expected_count: int) -> list[float]:
    return extract_scores(prediction, expected_count)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype == np.uint8:
        return values
    if np.issubdtype(values.dtype, np.floating):
        finite = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        if finite.max(initial=0.0) <= 1.0:
            finite = finite * 255.0
        return np.clip(finite, 0, 255).astype(np.uint8)
    return np.clip(values, 0, 255).astype(np.uint8)
