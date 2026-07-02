from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ImageInput = str | Path | Image.Image | np.ndarray


@dataclass(frozen=True)
class FaceDetectorConfig:
    """Configuration for the OpenCV Haar frontal-face detector.

    The detector returns bounding boxes as inclusive/exclusive pixel corners:
    x1/y1 are the top-left corner, x2/y2 are the first pixel outside the
    bottom-right corner. This matches PIL crop coordinates.
    """

    confidence_threshold: float = 0.50
    minimum_face_size: int = 32
    scale_factor: float = 1.08
    min_neighbors: int = 4


def load_rgb_image(image: ImageInput) -> Image.Image:
    """Load path/PIL/NumPy input as a PIL RGB image."""

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as loaded:
            return loaded.convert("RGB")

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim == 2:
            return Image.fromarray(_to_uint8(array), mode="L").convert("RGB")
        if array.ndim != 3:
            raise ValueError(f"Expected a 2D or 3D NumPy image array, got shape {array.shape}")
        if array.shape[2] == 1:
            return Image.fromarray(_to_uint8(array[:, :, 0]), mode="L").convert("RGB")
        if array.shape[2] == 3:
            return Image.fromarray(_to_uint8(array), mode="RGB")
        if array.shape[2] == 4:
            return Image.fromarray(_to_uint8(array), mode="RGBA").convert("RGB")
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {array.shape}")

    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def clamp_box(box: dict[str, Any], image_width: int, image_height: int) -> dict[str, int]:
    """Clamp x1/y1/x2/y2 to image bounds and return integer coordinates."""

    x1 = max(0, min(image_width, int(round(float(box["x1"])))))
    y1 = max(0, min(image_height, int(round(float(box["y1"])))))
    x2 = max(0, min(image_width, int(round(float(box["x2"])))))
    y2 = max(0, min(image_height, int(round(float(box["y2"])))))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def is_valid_box(box: dict[str, int], minimum_size: int = 1) -> bool:
    return (box["x2"] - box["x1"]) >= minimum_size and (box["y2"] - box["y1"]) >= minimum_size


def sort_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort detections deterministically from left to right, then top to bottom."""

    sorted_faces = sorted(detections, key=lambda face: (face["x1"], face["y1"], face["x2"], face["y2"]))
    for index, face in enumerate(sorted_faces):
        face["face_index"] = index
    return sorted_faces


class FaceDetector:
    """Reusable OpenCV face detector for local Phase 2 prototyping.

    This implementation uses the Haar cascade bundled with opencv-python. Haar
    does not emit calibrated probabilities, so confidence is a normalized form
    of OpenCV's cascade reject weight. Treat it as a detector ranking signal,
    not a real-world confidence value.
    """

    def __init__(self, config: FaceDetectorConfig | None = None, cascade_path: str | Path | None = None) -> None:
        self.config = config or FaceDetectorConfig()
        if not 0.0 <= self.config.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.config.minimum_face_size < 1:
            raise ValueError("minimum_face_size must be >= 1")
        if self.config.scale_factor <= 1.0:
            raise ValueError("scale_factor must be > 1.0")

        path = Path(cascade_path) if cascade_path else Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade_path = path
        self._cascade = cv2.CascadeClassifier(str(path))
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV Haar cascade: {path}")

    def detect(self, image: ImageInput) -> list[dict[str, Any]]:
        pil_image = load_rgb_image(image)
        rgb = np.asarray(pil_image)
        image_height, image_width = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        try:
            boxes, _reject_levels, reject_weights = self._cascade.detectMultiScale3(
                gray,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(self.config.minimum_face_size, self.config.minimum_face_size),
                outputRejectLevels=True,
            )
        except cv2.error as exc:
            raise RuntimeError(f"OpenCV face detection failed: {exc}") from exc

        boxes_array = np.asarray(boxes, dtype=float).reshape(-1, 4) if len(boxes) else np.empty((0, 4), dtype=float)
        weights = np.asarray(reject_weights, dtype=float).reshape(-1) if len(reject_weights) else np.zeros((len(boxes_array),))

        detections: list[dict[str, Any]] = []
        for raw_index, (x, y, w, h) in enumerate(boxes_array):
            confidence = _normalize_reject_weight(float(weights[raw_index])) if raw_index < len(weights) else 1.0
            box = clamp_box({"x1": x, "y1": y, "x2": x + w, "y2": y + h}, image_width, image_height)
            if confidence < self.config.confidence_threshold:
                continue
            if not is_valid_box(box, minimum_size=self.config.minimum_face_size):
                continue

            detections.append(
                {
                    "face_index": raw_index,
                    "x1": box["x1"],
                    "y1": box["y1"],
                    "x2": box["x2"],
                    "y2": box["y2"],
                    "confidence": round(float(confidence), 6),
                    "raw_detector_score": round(float(weights[raw_index]), 6) if raw_index < len(weights) else None,
                    "image_width": int(image_width),
                    "image_height": int(image_height),
                    "box_format": "xyxy_exclusive",
                }
            )

        return sort_detections(detections)


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


def _normalize_reject_weight(weight: float) -> float:
    if not math.isfinite(weight):
        return 0.0
    return float(np.clip(1.0 - math.exp(-max(weight, 0.0) / 4.0), 0.0, 1.0))
