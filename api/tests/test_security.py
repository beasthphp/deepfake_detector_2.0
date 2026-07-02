from __future__ import annotations

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.config import APIConfig
from api.exceptions import ServiceUnavailableError
from api.main import create_app
from api.services.inference_service import InferenceService


class SlowDetector:
    def __init__(self, delay: float, detections=None) -> None:
        self.delay = delay
        self.detections = detections or []

    def detect(self, image):
        time.sleep(self.delay)
        return self.detections


class FakeClassifier:
    def predict_images(self, images):
        return [{"real_score": 0.7, "fake_score": 0.3, "label": "Likely Real"} for _ in images]


class SlowService:
    def health_payload(self):
        return {"status": "ok", "model_loaded": True, "detector_loaded": True, "device": "CPU", "model_version": "test"}

    def model_info_payload(self):
        return {}

    def predict(self, image, image_info, decode_ms, request_id):
        time.sleep(0.25)
        return {
            "request_id": request_id,
            "status": "no_face_detected",
            "image": image_info,
            "faces_detected": 0,
            "faces": [],
            "timing_ms": {
                "decode": decode_ms,
                "face_detection": 0.0,
                "crop_preprocessing": 0.0,
                "classification": 0.0,
                "serialization": 0.0,
                "total": decode_ms,
            },
            "warnings": [],
        }


class ExplodingService(SlowService):
    def predict(self, image, image_info, decode_ms, request_id):
        raise RuntimeError("secret path D:\\ddp\\model\\deepfake_detector_93acc.h5")


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(20, 120, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_concurrent_requests_within_limit_succeed() -> None:
    service = InferenceService(
        APIConfig(request_concurrency_limit=2),
        detector=SlowDetector(0.05, []),
        classifier=FakeClassifier(),
    )
    service.model_loaded = True
    service.detector_loaded = True

    def call_once():
        return service.predict(Image.new("RGB", (64, 64)), {"width": 64, "height": 64, "format": "JPEG"}, 0.0, "x")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: call_once(), range(2)))

    assert [result["status"] for result in results] == ["no_face_detected", "no_face_detected"]


def test_concurrency_limit_rejects_excess_request() -> None:
    service = InferenceService(
        APIConfig(request_concurrency_limit=1),
        detector=SlowDetector(0.2, []),
        classifier=FakeClassifier(),
    )
    service.model_loaded = True
    service.detector_loaded = True
    ready = threading.Event()

    def first_call():
        ready.set()
        return service.predict(Image.new("RGB", (64, 64)), {"width": 64, "height": 64, "format": "JPEG"}, 0.0, "first")

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(first_call)
        ready.wait(timeout=1)
        time.sleep(0.02)
        with pytest.raises(ServiceUnavailableError) as exc_info:
            service.predict(Image.new("RGB", (64, 64)), {"width": 64, "height": 64, "format": "JPEG"}, 0.0, "second")
        future.result(timeout=2)

    assert exc_info.value.status_code == 429
    assert exc_info.value.code == "concurrency_limit_exceeded"


def test_request_timeout_returns_504() -> None:
    config = APIConfig(request_timeout_seconds=0.05)
    with TestClient(create_app(config, service=SlowService())) as client:
        response = client.post("/predict", files={"file": ("photo.jpg", image_bytes(), "image/jpeg")})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "request_timeout"


def test_internal_errors_are_sanitized() -> None:
    with TestClient(create_app(APIConfig(), service=ExplodingService()), raise_server_exceptions=False) as client:
        response = client.post("/predict", files={"file": ("photo.jpg", image_bytes(), "image/jpeg")})
    assert response.status_code == 500
    assert "D:\\ddp" not in response.text
    assert response.json()["error"]["code"] == "internal_server_error"
