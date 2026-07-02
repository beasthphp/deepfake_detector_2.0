# Human-Face Deepfake Detection MVP

This repository contains an experimental MVP for human-face deepfake detection on local images and user-controlled webpage images.

The included architecture is functional, but the current model is an experimental placeholder and should not be treated as proof of authenticity.

## Purpose

The system detects human faces, crops each detected face, classifies each crop with a replaceable local model provider, and returns stable metadata that a browser extension can render as per-face overlays.

## Architecture

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

## Completed Features

- Existing TensorFlow human-face classifier audit and evaluation.
- Face detection and crop pipeline.
- Local FastAPI inference service.
- Manual extension image analysis.
- User-controlled visible webpage scanning.
- Request queue, caching, and deduplication.
- Responsive per-face overlays.
- API and extension automated tests.
- Replaceable model-provider boundary.

## Quick Start

Create an environment and install local API dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-api.txt
```

Place the current legacy model at:

```text
models/artifacts/deepfake_detector_93acc.h5
```

Or set an explicit local override:

```powershell
$env:DEEPFAKE_MODEL_PATH="model/deepfake_detector_93acc.h5"
python scripts/verify_model.py --model-id legacy-cnn-v1
```

Start the API:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Extension Installation

1. Open Chrome or Chromium extension management.
2. Enable developer mode.
3. Load the `extension/` directory as an unpacked extension.
4. Keep the local API running at `http://127.0.0.1:8000`, unless extension mock mode is enabled.

## Usage

Manual analysis:

- Right-click a webpage image.
- Choose the extension analysis action.
- Review the popup or results page.

Visible-page scanning:

- Open the extension popup.
- Choose `Scan visible images`.
- Review overlays on detected faces.
- Use stop, rescan, clear results, emergency stop, and per-site controls from the popup.

## Test Commands

```powershell
python -m compileall api face_pipeline model_runtime scripts
pytest api/tests face_pipeline/test_pipeline.py
npm run test:extension
node --check extension/content-script.js
python scripts/scan_extension_safety.py extension
```

Actual-model tests are skipped when the model artifact is unavailable. Ordinary CI does not require datasets, a GPU, or real model files.

## Model Installation

Models are selected by `ACTIVE_MODEL_ID` from `models/registry.json`. The default ID is `legacy-cnn-v1`.

The registry supports:

- local filename
- optional download URL
- optional SHA-256
- provider name
- input/output contract
- class mapping
- threshold profile
- enabled status

Use:

```powershell
python scripts/download_model.py --model-id legacy-cnn-v1
python scripts/verify_model.py --model-id legacy-cnn-v1
```

The current registry has no download URL, so developers must place the model manually or set `DEEPFAKE_MODEL_PATH`.

## Privacy Behavior

The extension sends only user-selected images or eligible images from user-controlled scans. With default settings, images are sent only to the local API. Raw image bytes and face crops are not stored by the extension cache.

If a remote API URL is configured, eligible images are transmitted to that remote origin.

## Limitations

- The current model is an experimental placeholder.
- Real-world webpage predictions are unreliable.
- No external holdout validation has been completed.
- Scores are uncalibrated model outputs.
- Haar face detection misses some valid faces and works best on frontal visible faces.
- Manual browser testing for Phase 5 remains incomplete.
- The local API must be running for non-mock extension use.

## Roadmap

Immediate next work focuses on completing manual browser testing, collecting multiple datasets, using identity-aware or source-aware splits, training a MobileNetV3Large baseline, externally evaluating replacement models, and calibrating uncertainty thresholds.

## Repository Structure

```text
api/                FastAPI service and API tests
extension/          Manifest V3 browser extension
face_pipeline/      Face detection, crop, and local pipeline utilities
model_runtime/      Replaceable model-provider boundary
models/             Model registry and artifact instructions
scripts/            Model and validation helper scripts
reports/            Completed audit, phase, and release-prep reports
training/           Experimental training utilities
evaluation/         Existing-model evaluation script
```
