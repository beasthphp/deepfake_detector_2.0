# Architecture

```text
Webpage image
    -> Extension eligibility and queue
    -> Local API
    -> Face detector
    -> Crop strategy
    -> Model provider
    -> Stable response contract
    -> Overlay renderer
```

## Extension

The Manifest V3 extension supports manual image analysis and user-controlled visible-page scanning. It filters eligible images, deduplicates requests, sends images to the configured API origin, caches results by model/detector/crop/threshold version signature, and renders per-face overlays against displayed image geometry.

## Local API

The FastAPI service receives one uploaded image, decodes it safely, enforces upload and image limits, detects faces, crops each face, classifies crops in a batch, and returns a stable JSON response.

The `/predict` contract exposes:

- model ID and version
- detector ID and version
- crop-strategy ID and version
- threshold ID and version
- per-face bounding boxes, crop boxes, scores, and labels
- timing and warning metadata

## Face Pipeline

The face pipeline owns detector and crop behavior. It does not decide whether a model is TensorFlow, ONNX, TFLite, or another framework. Detector failure and crop failure are never converted into a real prediction.

## Replaceable-Model Boundary

Model-specific behavior lives under `model_runtime/providers/`.

Only a provider should know:

- framework imports
- model file type
- input shape and preprocessing
- output shape and score interpretation
- hash and artifact validation

The registry in `models/registry.json` selects the active model by `ACTIVE_MODEL_ID`. `DEEPFAKE_MODEL_PATH` can override the local artifact path without changing source code.

Startup fails clearly when the active model is unknown, disabled, missing, a Git LFS pointer, hash-mismatched, provider-unavailable, or contract-incompatible.
