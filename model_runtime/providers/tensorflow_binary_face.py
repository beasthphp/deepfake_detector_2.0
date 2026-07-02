from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from model_runtime.base import DeepfakeModelProvider
from model_runtime.exceptions import ModelArtifactError, ModelContractError, ModelHashError, ProviderUnavailableError
from model_runtime.metadata import ModelMetadata, ScoreThresholds, ThresholdProfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class TensorFlowBinaryFaceProvider(DeepfakeModelProvider):
    def __init__(self, entry: Any, model_path: Path, model: Any | None = None) -> None:
        self.entry = entry
        self.model_path = Path(model_path)
        self._model = model
        self._metadata = entry.metadata()

    @classmethod
    def from_loaded_model(
        cls,
        model: Any,
        thresholds: ScoreThresholds | None = None,
        model_id: str = "test-model",
        model_version: str = "test",
    ) -> "TensorFlowBinaryFaceProvider":
        threshold_profile = ThresholdProfile(thresholds=thresholds or ScoreThresholds())

        class LoadedEntry:
            local_filename = "<in-memory>"
            sha256 = None
            file_types = (".h5", ".keras")

            def metadata(self) -> ModelMetadata:
                return ModelMetadata(
                    id=model_id,
                    version=model_version,
                    framework="tensorflow-keras",
                    provider="tensorflow_binary_face",
                    input_shape=(None, 256, 256, 3),
                    output_shape=(None, 1),
                    output_meaning="Single sigmoid score where real_score = model output and fake_score = 1 - real_score.",
                    class_mapping={"fake": 0, "real": 1},
                    threshold_profile=threshold_profile,
                    model_name="in-memory test classifier",
                )

        return cls(entry=LoadedEntry(), model_path=Path("<in-memory>"), model=model)

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    def load(self) -> None:
        self.validate_file()
        try:
            from tensorflow.keras.models import load_model
        except Exception as exc:
            raise ProviderUnavailableError("TensorFlow/Keras is required for provider tensorflow_binary_face.") from exc

        try:
            self._model = load_model(self.model_path, compile=False)
        except Exception as exc:
            raise ModelArtifactError(f"Model could not be loaded: {self.model_path}") from exc
        self.validate_contract()

    def validate(self) -> None:
        if self._model is None:
            self.load()
        else:
            self.validate_contract()

    def validate_file(self) -> None:
        path = self.model_path
        if not path.exists():
            raise ModelArtifactError(f"Model file is missing: {path}")
        if path.suffix.lower() not in self.entry.file_types:
            raise ModelArtifactError(f"Model file has unexpected type: {path.suffix}")
        with path.open("rb") as handle:
            header = handle.read(256)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1") or header.startswith(b"version "):
            raise ModelArtifactError(f"Model file is a Git LFS pointer, not a model: {path}")
        if path.suffix.lower() == ".h5" and not header.startswith(b"\x89HDF"):
            raise ModelArtifactError(f"Model file does not look like HDF5/Keras data: {path}")
        if self.entry.sha256:
            actual = sha256_file(path)
            if actual.lower() != str(self.entry.sha256).lower():
                raise ModelHashError(f"Model SHA-256 mismatch for {path}")

    def validate_contract(self) -> None:
        model = self.model
        input_shape = normalize_shape(getattr(model, "input_shape", None))
        output_shape = normalize_shape(getattr(model, "output_shape", None))
        if input_shape != self.metadata.input_shape:
            raise ModelContractError(f"Unexpected model input shape: {input_shape}; expected {self.metadata.input_shape}")
        if output_shape != self.metadata.output_shape:
            raise ModelContractError(f"Unexpected model output shape: {output_shape}; expected {self.metadata.output_shape}")

    def predict_batch(self, images: list[Any]) -> list[dict[str, float | str]]:
        if not images:
            return []
        batch = np.concatenate([preprocess_binary_face_image(image, self.metadata.input_shape) for image in images], axis=0)
        predictions = self.model.predict(batch, verbose=0)
        scores = extract_scores(predictions, expected_count=len(images))
        thresholds = self.metadata.threshold_profile.thresholds
        return [
            {
                "real_score": float(score),
                "fake_score": float(1.0 - score),
                "label": label_from_real_score(float(score), thresholds),
            }
            for score in scores
        ]


def preprocess_binary_face_image(image: Image.Image | np.ndarray, input_shape: tuple[int | None, ...]) -> np.ndarray:
    if len(input_shape) != 4 or input_shape[1:] != (256, 256, 3):
        raise ModelContractError(f"Unsupported provider input shape for preprocessing: {input_shape}")

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            pil_image = Image.fromarray(_to_uint8(array), mode="L").convert("RGB")
        elif array.ndim == 3 and array.shape[2] == 1:
            pil_image = Image.fromarray(_to_uint8(array[:, :, 0]), mode="L").convert("RGB")
        elif array.ndim == 3 and array.shape[2] == 3:
            pil_image = Image.fromarray(_to_uint8(array), mode="RGB")
        elif array.ndim == 3 and array.shape[2] == 4:
            pil_image = Image.fromarray(_to_uint8(array), mode="RGBA").convert("RGB")
        else:
            raise ValueError(f"Unsupported image array shape for preprocessing: {array.shape}")

    resampling = getattr(Image, "Resampling", Image).NEAREST
    resized = pil_image.resize((256, 256), resample=resampling)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    if values.shape != (256, 256, 3):
        raise RuntimeError(f"Unexpected preprocessed image shape: {values.shape}")
    return np.expand_dims(values, axis=0)


def label_from_real_score(real_score: float, thresholds: ScoreThresholds | None = None) -> str:
    config = thresholds or ScoreThresholds()
    if real_score < config.likely_fake_below:
        return "Likely Fake"
    if real_score > config.likely_real_above:
        return "Likely Real"
    return "Uncertain"


def extract_single_score(prediction: Any) -> float:
    return extract_scores(prediction, expected_count=1)[0]


def extract_scores(prediction: Any, expected_count: int) -> list[float]:
    values = np.asarray(prediction, dtype=np.float32).reshape(-1)
    if values.size != expected_count:
        raise ValueError(f"Expected {expected_count} model score(s), got shape {np.asarray(prediction).shape}")
    scores: list[float] = []
    for value in values:
        score = float(value)
        if not np.isfinite(score):
            raise ValueError(f"Model output is not finite: {score}")
        if score < 0.0 or score > 1.0:
            raise ValueError(f"Model output should be in [0, 1], got {score}")
        scores.append(score)
    return scores


def ensure_real_hdf5(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        raise ModelArtifactError(f"Model file is missing: {path}")
    with path.open("rb") as handle:
        header = handle.read(256)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1") or header.startswith(b"version "):
        raise ModelArtifactError(f"Model file is a Git LFS pointer, not a model: {path}")
    if not header.startswith(b"\x89HDF"):
        raise ModelArtifactError(f"Model file does not look like HDF5/Keras data: {path}")


def normalize_shape(value: Any) -> tuple[int | None, ...]:
    if value is None:
        return ()
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return tuple(None if item is None else int(item) for item in value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
