# Local inference API

This directory contains the FastAPI service that exposes the deepfake-analysis pipeline to local clients and the browser extension.

The API is responsible for request validation, image decoding, face-pipeline execution, model inference, and returning a stable per-face response contract. Model-specific logic remains outside the HTTP layer so the detector can be replaced without redesigning the API.
