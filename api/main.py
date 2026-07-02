from __future__ import annotations

import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import APIConfig
from api.dependencies import get_inference_service
from api.exceptions import APIError, InferenceProcessingError
from api.schemas import HealthResponse, ModelInfoResponse, PredictionResponse
from api.security import read_and_decode_upload
from api.services.inference_service import InferenceService


ServiceFactory = Callable[[APIConfig], Any]


def create_app(
    config: APIConfig | None = None,
    service: Any | None = None,
    service_factory: ServiceFactory | None = None,
) -> FastAPI:
    settings = config or APIConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.inference_service = service
        else:
            factory = service_factory or InferenceService.load
            app.state.inference_service = factory(settings)
        app.state.config = settings
        yield

    api = FastAPI(
        title="Human-Face Deepfake Detector API",
        version="0.1.0",
        lifespan=lifespan,
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @api.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "request_id": getattr(request.state, "request_id", None),
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                },
            },
        )

    @api.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "request_id": getattr(request.state, "request_id", None),
                "error": {
                    "code": "internal_server_error",
                    "message": "The server could not complete the request.",
                },
            },
        )

    @api.get("/health", response_model=HealthResponse)
    async def health(inference_service: InferenceService = Depends(get_inference_service)) -> dict[str, Any]:
        return inference_service.health_payload()

    @api.get("/model-info", response_model=ModelInfoResponse)
    async def model_info(inference_service: InferenceService = Depends(get_inference_service)) -> dict[str, Any]:
        return inference_service.model_info_payload()

    @api.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        inference_service: InferenceService = Depends(get_inference_service),
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        image, image_info, decode_ms = await read_and_decode_upload(file, settings)
        try:
            with anyio.fail_after(settings.request_timeout_seconds):
                return await anyio.to_thread.run_sync(
                    inference_service.predict,
                    image,
                    image_info,
                    decode_ms,
                    request_id,
                    abandon_on_cancel=True,
                )
        except TimeoutError as exc:
            raise InferenceProcessingError(504, "request_timeout", "Inference request timed out.") from exc

    return api


app = create_app()
