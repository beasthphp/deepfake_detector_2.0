from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.config import APIConfig
from api.main import create_app
from api.services.inference_service import InferenceService


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "phase2_samples" / "input"
LEGACY_MODEL_PATH = ROOT / "model" / "deepfake_detector_93acc.h5"


@pytest.fixture(scope="module")
def actual_client():
    if not LEGACY_MODEL_PATH.exists():
        pytest.skip("Actual model artifact is not available.")
    try:
        with TestClient(create_app(APIConfig(model_path=LEGACY_MODEL_PATH))) as client:
            yield client
    except Exception as exc:
        pytest.skip(f"Actual model startup failed: {exc}")


class FakeDetector:
    def __init__(self, detections, exc: Exception | None = None) -> None:
        self.detections = detections
        self.exc = exc

    def detect(self, image):
        if self.exc:
            raise self.exc
        return self.detections


class FakeClassifier:
    def __init__(self, scores=None, exc: Exception | None = None) -> None:
        self.scores = scores or [0.75]
        self.exc = exc
        self.calls = 0

    def predict_images(self, images):
        self.calls += 1
        if self.exc:
            raise self.exc
        return [
            {
                "real_score": float(score),
                "fake_score": float(1.0 - score),
                "label": "Likely Real" if score > 0.60 else "Likely Fake",
            }
            for score in self.scores[: len(images)]
        ]


def post_sample(client: TestClient, filename: str):
    with (SAMPLE_DIR / filename).open("rb") as handle:
        return client.post("/predict", files={"file": (filename, handle.read(), "image/jpeg")})


def test_predict_clear_real_runs_actual_model(actual_client: TestClient) -> None:
    response = post_sample(actual_client, "one_clear_real.jpg")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["faces_detected"] == 1
    assert 0.0 <= payload["faces"][0]["real_score"] <= 1.0


def test_predict_clear_fake_runs_actual_model(actual_client: TestClient) -> None:
    response = post_sample(actual_client, "one_clear_fake.jpg")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["faces_detected"] == 1
    assert payload["faces"][0]["fake_score"] == pytest.approx(1.0 - payload["faces"][0]["real_score"])


def test_predict_multiple_faces_runs_actual_model(actual_client: TestClient) -> None:
    response = post_sample(actual_client, "multiple_faces_real_fake.jpg")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["faces_detected"] >= 2
    assert [face["face_index"] for face in payload["faces"]] == list(range(len(payload["faces"])))


def test_no_face_is_not_likely_real(actual_client: TestClient) -> None:
    response = post_sample(actual_client, "no_face_geometric.jpg")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "no_face_detected"
    assert payload["faces_detected"] == 0
    assert payload["faces"] == []
    assert payload["warnings"]


def test_face_near_boundary_is_processed_with_context_crop() -> None:
    detection = {
        "face_index": 0,
        "x1": 1,
        "y1": 1,
        "x2": 52,
        "y2": 60,
        "confidence": 0.9,
        "image_width": 128,
        "image_height": 128,
    }
    service = InferenceService(APIConfig(), detector=FakeDetector([detection]), classifier=FakeClassifier([0.7]))
    service.model_loaded = True
    service.detector_loaded = True
    result = service.predict(Image.new("RGB", (128, 128)), {"width": 128, "height": 128, "format": "JPEG"}, 0.0, "x")
    assert result["status"] == "completed"
    assert result["faces"][0]["preserved_original"] is False
    assert result["faces"][0]["crop_strategy"] == "context_full_head_l40_r40_t70_b35_square"


def test_multiple_faces_are_classified_in_one_batch() -> None:
    detections = [
        {"face_index": 0, "x1": 20, "y1": 20, "x2": 80, "y2": 90, "confidence": 0.9, "image_width": 160, "image_height": 160},
        {"face_index": 1, "x1": 90, "y1": 30, "x2": 150, "y2": 100, "confidence": 0.8, "image_width": 160, "image_height": 160},
    ]
    classifier = FakeClassifier([0.2, 0.8])
    service = InferenceService(APIConfig(), detector=FakeDetector(detections), classifier=classifier)
    service.model_loaded = True
    service.detector_loaded = True
    result = service.predict(Image.new("RGB", (160, 160)), {"width": 160, "height": 160, "format": "PNG"}, 0.0, "x")
    assert result["faces_detected"] == 2
    assert classifier.calls == 1


def test_too_many_faces_returns_422() -> None:
    detections = [
        {"face_index": i, "x1": i, "y1": 10, "x2": i + 40, "y2": 50, "confidence": 0.9, "image_width": 128, "image_height": 128}
        for i in range(2)
    ]
    service = InferenceService(APIConfig(max_faces_per_image=1), detector=FakeDetector(detections), classifier=FakeClassifier())
    service.model_loaded = True
    service.detector_loaded = True
    with pytest.raises(Exception) as exc_info:
        service.predict(Image.new("RGB", (128, 128)), {"width": 128, "height": 128, "format": "JPEG"}, 0.0, "x")
    assert getattr(exc_info.value, "status_code") == 422


def test_detector_exception_is_not_translated_to_real() -> None:
    service = InferenceService(APIConfig(), detector=FakeDetector([], exc=RuntimeError("detector boom")), classifier=FakeClassifier())
    service.model_loaded = True
    service.detector_loaded = True
    with pytest.raises(Exception) as exc_info:
        service.predict(Image.new("RGB", (128, 128)), {"width": 128, "height": 128, "format": "JPEG"}, 0.0, "x")
    assert getattr(exc_info.value, "status_code") == 500
    assert getattr(exc_info.value, "code") == "detector_failed"


def test_classifier_exception_is_not_translated_to_real() -> None:
    detection = {"face_index": 0, "x1": 20, "y1": 20, "x2": 90, "y2": 95, "confidence": 0.9, "image_width": 128, "image_height": 128}
    service = InferenceService(APIConfig(), detector=FakeDetector([detection]), classifier=FakeClassifier(exc=RuntimeError("classifier boom")))
    service.model_loaded = True
    service.detector_loaded = True
    with pytest.raises(Exception) as exc_info:
        service.predict(Image.new("RGB", (128, 128)), {"width": 128, "height": 128, "format": "JPEG"}, 0.0, "x")
    assert getattr(exc_info.value, "status_code") == 500
    assert getattr(exc_info.value, "code") == "classifier_failed"
