from __future__ import annotations

from fastapi.testclient import TestClient

from api.config import APIConfig
from api.main import create_app


class StaticService:
    def health_payload(self):
        return {
            "status": "ok",
            "model_loaded": True,
            "detector_loaded": True,
            "device": "CPU",
            "model_version": "existing-cnn-93acc",
        }

    def model_info_payload(self):
        return {
            "model_name": "existing human-face CNN deepfake detector",
            "model_version": "existing-cnn-93acc",
            "input_shape": [None, 256, 256, 3],
            "output_shape": [None, 1],
            "output_meaning": "real_score = model output; fake_score = 1 - real_score",
            "class_mapping": {"fake": 0, "real": 1},
            "selected_face_detector": "opencv_haar_frontalface_default",
            "selected_crop_strategy": "preserve_or_context_full_head_l40_r40_t70_b35_square",
            "provisional_thresholds": {
                "likely_fake": "real_score < 0.40",
                "uncertain": "0.40 <= real_score <= 0.60",
                "likely_real": "real_score > 0.60",
            },
            "known_limitations": ["CPU-only local MVP"],
        }


def test_health_endpoint() -> None:
    with TestClient(create_app(APIConfig(), service=StaticService())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["detector_loaded"] is True
    assert payload["device"] == "CPU"


def test_model_info_endpoint_does_not_expose_paths() -> None:
    with TestClient(create_app(APIConfig(), service=StaticService())) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    text = response.text
    assert "D:\\ddp" not in text
    assert "deepfake_detector_93acc.h5" not in text
    payload = response.json()
    assert payload["class_mapping"] == {"fake": 0, "real": 1}
    assert payload["selected_face_detector"] == "opencv_haar_frontalface_default"
