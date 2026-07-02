from __future__ import annotations

from fastapi import Request

from api.services.inference_service import InferenceService


def get_inference_service(request: Request) -> InferenceService:
    return request.app.state.inference_service
