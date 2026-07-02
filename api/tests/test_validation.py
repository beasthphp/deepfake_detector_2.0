from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from api.config import APIConfig
from api.main import create_app


class NoFaceService:
    def health_payload(self):
        return {"status": "ok", "model_loaded": True, "detector_loaded": True, "device": "CPU", "model_version": "test"}

    def model_info_payload(self):
        return {
            "model_name": "test",
            "model_version": "test",
            "input_shape": [None, 256, 256, 3],
            "output_shape": [None, 1],
            "output_meaning": "test",
            "class_mapping": {"fake": 0, "real": 1},
            "selected_face_detector": "test",
            "selected_crop_strategy": "preserve_or_context_full_head_l40_r40_t70_b35_square",
            "provisional_thresholds": {"likely_fake": "x", "uncertain": "x", "likely_real": "x"},
            "known_limitations": [],
        }

    def predict(self, image, image_info, decode_ms, request_id):
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
            "warnings": ["No supported human face was detected. The deepfake classifier was not run."],
        }


def client(config: APIConfig | None = None) -> TestClient:
    return TestClient(create_app(config or APIConfig(), service=NoFaceService()))


def image_bytes(size=(64, 64), fmt="JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(200, 80, 40)).save(buffer, format=fmt)
    return buffer.getvalue()


def post_file(test_client: TestClient, content: bytes, filename: str, content_type: str):
    return test_client.post("/predict", files={"file": (filename, content, content_type)})


def test_rejects_tiny_image() -> None:
    with client() as test_client:
        response = post_file(test_client, image_bytes((16, 16)), "tiny.jpg", "image/jpeg")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "image_too_small"


def test_rejects_oversized_upload_before_decode() -> None:
    config = APIConfig(max_upload_bytes=16)
    with client(config) as test_client:
        response = post_file(test_client, b"x" * 17, "large.jpg", "image/jpeg")
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"


def test_rejects_unsupported_text_content_type() -> None:
    with client() as test_client:
        response = post_file(test_client, b"hello", "note.txt", "text/plain")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_content_type"


def test_rejects_corrupted_jpeg() -> None:
    with client() as test_client:
        response = post_file(test_client, b"\xff\xd8not a jpeg", "bad.jpg", "image/jpeg")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "malformed_image"


def test_valid_image_with_incorrect_extension_is_accepted() -> None:
    with client() as test_client:
        response = post_file(test_client, image_bytes(), "photo.txt", "image/jpeg")
    assert response.status_code == 200
    assert response.json()["status"] == "no_face_detected"


def test_valid_extension_with_non_image_data_is_rejected() -> None:
    with client() as test_client:
        response = post_file(test_client, b"not image bytes", "photo.jpg", "image/jpeg")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "malformed_image"


def test_rejects_empty_file() -> None:
    with client() as test_client:
        response = post_file(test_client, b"", "empty.jpg", "image/jpeg")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_upload"
