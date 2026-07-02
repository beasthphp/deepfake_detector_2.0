from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from PIL import Image

from api.config import APIConfig
from api.exceptions import InferenceProcessingError, ServiceUnavailableError
from face_pipeline.classifier import DeepfakeClassifier
from face_pipeline.cropper import (
    SELECTED_CROP_STRATEGY,
    crop_face_selected_strategy,
)
from face_pipeline.detector import FaceDetector, FaceDetectorConfig
from model_runtime.metadata import ModelMetadata, ScoreThresholds


WARNINGS = [
    "Results are probabilistic and are not proof of authenticity.",
    "No-face or detector failures are never interpreted as Likely Real.",
]


class InferenceService:
    def __init__(
        self,
        config: APIConfig,
        detector: FaceDetector | None = None,
        classifier: DeepfakeClassifier | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.classifier = classifier
        self.model_loaded = False
        self.detector_loaded = False
        self.input_shape: tuple[Any, ...] | None = None
        self.output_shape: tuple[Any, ...] | None = None
        self.device = "CPU"
        self._request_slots = threading.BoundedSemaphore(config.request_concurrency_limit)
        self._inference_lock = threading.Lock()

    @classmethod
    def load(cls, config: APIConfig) -> "InferenceService":
        service = cls(config=config)
        service.startup_validate()
        return service

    def startup_validate(self) -> None:
        self.classifier = DeepfakeClassifier(
            model_path=self.config.model_path,
            thresholds=self.config.thresholds,
            active_model_id=self.config.active_model_id,
            registry_path=self.config.model_registry_path,
            artifact_dir=self.config.model_artifact_dir,
        )
        self.classifier.provider.validate()
        metadata = self.classifier.metadata
        self.input_shape = metadata.input_shape
        self.output_shape = metadata.output_shape
        self.model_loaded = True

        self.detector = FaceDetector(FaceDetectorConfig())
        self.detector_loaded = True

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.model_loaded and self.detector_loaded else "starting",
            "model_loaded": self.model_loaded,
            "detector_loaded": self.detector_loaded,
            "device": self.device,
            "model_id": self._model_id(),
            "model_version": self._model_version(),
            "detector_id": self.config.detector_id,
            "detector_version": self.config.detector_version,
            "crop_strategy_id": self.config.selected_crop_strategy,
            "crop_strategy_version": self.config.crop_strategy_version,
            "threshold_id": self._threshold_id(),
            "threshold_version": self._threshold_version(),
        }

    def model_info_payload(self) -> dict[str, Any]:
        metadata = self._metadata()
        input_shape = list(self.input_shape or (None, 256, 256, 3))
        output_shape = list(self.output_shape or (None, 1))
        thresholds = self._thresholds()
        return {
            "model_name": metadata.model_name if metadata else "existing human-face CNN deepfake detector",
            "model_id": self._model_id(),
            "model_version": self._model_version(),
            "input_shape": input_shape,
            "output_shape": output_shape,
            "output_meaning": metadata.output_meaning if metadata else "Single sigmoid-like score where real_score = model output and fake_score = 1 - real_score.",
            "class_mapping": metadata.class_mapping if metadata else {"fake": 0, "real": 1},
            "selected_face_detector": self.config.detector_id,
            "selected_face_detector_version": self.config.detector_version,
            "selected_crop_strategy": SELECTED_CROP_STRATEGY,
            "selected_crop_strategy_version": self.config.crop_strategy_version,
            "threshold_version": self._threshold_version(),
            "provisional_thresholds": {
                "likely_fake": f"real_score < {thresholds.likely_fake_below:.2f}",
                "uncertain": f"{thresholds.likely_fake_below:.2f} <= real_score <= {thresholds.likely_real_above:.2f}",
                "likely_real": f"real_score > {thresholds.likely_real_above:.2f}",
            },
            "known_limitations": [
                "Haar cascade missed 5% of images in Phase 2.1 expanded validation.",
                "Haar works best on frontal, visible human faces.",
                "The classifier was trained on prepared face images, not arbitrary real-world scenes.",
                "Scores are uncalibrated model outputs, not proof of authenticity.",
                "This is a CPU-only local MVP and not a high-scalability deployment.",
                "Cold startup is affected by the large existing .h5 model.",
            ],
        }

    def predict(self, image: Image.Image, image_info: dict[str, Any], decode_ms: float, request_id: str) -> dict[str, Any]:
        if self.detector is None or self.classifier is None:
            raise ServiceUnavailableError(503, "service_not_ready", "Inference service is not ready.")
        if not self._request_slots.acquire(blocking=False):
            raise ServiceUnavailableError(429, "concurrency_limit_exceeded", "Inference service is busy.")

        request_started = time.perf_counter()
        detection_ms = 0.0
        crop_ms = 0.0
        classification_ms = 0.0
        serialization_ms = 0.0
        try:
            with self._inference_lock:
                detection_started = time.perf_counter()
                try:
                    detections = self.detector.detect(image)
                except Exception as exc:
                    raise InferenceProcessingError(500, "detector_failed", "Face detection failed.") from exc
                detection_ms = elapsed_ms(detection_started)

                if len(detections) > self.config.max_faces_per_image:
                    raise InferenceProcessingError(
                        422,
                        "too_many_faces",
                        f"Detected more than {self.config.max_faces_per_image} faces.",
                    )

                if not detections:
                    total_ms = elapsed_ms(request_started) + decode_ms
                    return {
                        "request_id": request_id,
                        "status": "no_face_detected",
                        **self.prediction_metadata(),
                        "image": image_info,
                        "faces_detected": 0,
                        "faces": [],
                        "timing_ms": {
                            "decode": decode_ms,
                            "face_detection": detection_ms,
                            "crop_preprocessing": 0.0,
                            "classification": 0.0,
                            "serialization": 0.0,
                            "total": round(total_ms, 3),
                        },
                        "warnings": [
                            "No supported human face was detected. The deepfake classifier was not run.",
                            "No-face results are never interpreted as Likely Real.",
                        ],
                    }

                crop_started = time.perf_counter()
                crop_results = []
                crops = []
                for detection in detections:
                    try:
                        crop_result = crop_face_selected_strategy(image, detection, detections)
                    except Exception as exc:
                        raise InferenceProcessingError(422, "crop_failed", "Detected face could not be cropped.") from exc
                    crop_results.append((detection, crop_result))
                    crops.append(crop_result.crop)
                crop_ms = elapsed_ms(crop_started)

                classification_started = time.perf_counter()
                try:
                    predictions = self.classifier.predict_images(crops)
                except Exception as exc:
                    raise InferenceProcessingError(500, "classifier_failed", "Deepfake classification failed.") from exc
                classification_ms = elapsed_ms(classification_started)

            serialization_started = time.perf_counter()
            faces = []
            for face_index, ((detection, crop_result), prediction) in enumerate(zip(crop_results, predictions)):
                real_score = float(prediction["real_score"])
                fake_score = float(prediction["fake_score"])
                validate_scores(real_score, fake_score)
                faces.append(
                    {
                        "face_index": face_index,
                        "bounding_box": public_box(detection),
                        "crop_box": crop_result.crop_box,
                        "face_detection_score": float(detection["confidence"]) if detection.get("confidence") is not None else None,
                        "crop_strategy": crop_result.crop_strategy,
                        "preserved_original": crop_result.preserved_original,
                        "real_score": real_score,
                        "fake_score": fake_score,
                        "label": str(prediction["label"]),
                    }
                )

            serialization_ms = elapsed_ms(serialization_started)
            total_ms = elapsed_ms(request_started) + decode_ms
            return {
                "request_id": request_id,
                "status": "completed",
                **self.prediction_metadata(),
                "image": image_info,
                "faces_detected": len(detections),
                "faces": faces,
                "timing_ms": {
                    "decode": decode_ms,
                    "face_detection": detection_ms,
                    "crop_preprocessing": crop_ms,
                    "classification": classification_ms,
                    "serialization": serialization_ms,
                    "total": round(total_ms, 3),
                },
                "warnings": WARNINGS,
            }
        finally:
            self._request_slots.release()

    def prediction_metadata(self) -> dict[str, Any]:
        thresholds = self._thresholds()
        return {
            "model": {
                "id": self._model_id(),
                "version": self._model_version(),
            },
            "detector": {
                "id": self.config.detector_id,
                "version": self.config.detector_version,
            },
            "crop_strategy": {
                "id": self.config.selected_crop_strategy,
                "version": self.config.crop_strategy_version,
            },
            "thresholds": {
                "id": self._threshold_id(),
                "version": self._threshold_version(),
                "likely_fake_below": thresholds.likely_fake_below,
                "likely_real_above": thresholds.likely_real_above,
            },
        }

    def _metadata(self) -> ModelMetadata | None:
        if self.classifier is None:
            return None
        return getattr(self.classifier, "metadata", None)

    def _model_id(self) -> str:
        metadata = self._metadata()
        return metadata.id if metadata else self.config.model_id

    def _model_version(self) -> str:
        metadata = self._metadata()
        return metadata.version if metadata else self.config.model_version

    def _thresholds(self) -> ScoreThresholds:
        metadata = self._metadata()
        if metadata:
            return metadata.threshold_profile.thresholds
        return self.config.thresholds

    def _threshold_id(self) -> str:
        metadata = self._metadata()
        return metadata.threshold_profile.id if metadata else self.config.threshold_id

    def _threshold_version(self) -> str:
        metadata = self._metadata()
        return metadata.threshold_profile.version if metadata else self.config.threshold_version


def validate_scores(real_score: float, fake_score: float) -> None:
    values = [real_score, fake_score]
    if any(not np.isfinite(value) for value in values):
        raise InferenceProcessingError(500, "invalid_model_score", "Model returned a non-finite score.")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise InferenceProcessingError(500, "invalid_model_score", "Model returned a score outside [0, 1].")


def public_box(detection: dict[str, Any]) -> dict[str, int]:
    return {key: int(detection[key]) for key in ("x1", "y1", "x2", "y2")}


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)
