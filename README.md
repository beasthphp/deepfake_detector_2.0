# Human-Face Deepfake Detection — Browser Extension MVP

A local-first deepfake-detection prototype that finds human faces in webpage images, crops each face, runs a replaceable TensorFlow/Keras classifier, and renders per-face results through a Chrome Manifest V3 extension.

The system is designed as an **end-to-end ML product prototype**: face preprocessing, model inference, FastAPI serving, browser integration, queueing/caching, automated tests, and explicit model limitations are all separated into replaceable components.

> **Important:** the current classifier is an experimental baseline. Its predictions are not proof that an image is authentic or manipulated, and its output scores are not calibrated confidence values.

## System Flow

```text
Webpage image
      |
      v
Chrome MV3 Extension
      |
      +--> eligibility checks
      +--> queue / deduplication / cache
      |
      v
Local FastAPI Service
      |
      v
Face Detection
      |
      v
Face Crop + Preprocessing
      |
      v
Replaceable Model Provider
      |
      v
Likely Fake / Uncertain / Likely Real
      |
      v
Per-face Browser Overlay
```

## What Is Implemented

- human-face detection and crop pipeline
- local FastAPI inference service
- Chrome Manifest V3 browser extension
- right-click analysis of a selected webpage image
- user-controlled scanning of visible page images
- request queueing, caching, and image deduplication
- per-face prediction overlays
- site enable/disable controls and emergency-stop behavior
- stable API response contract between model runtime and extension
- replaceable model-provider boundary
- automated API, pipeline, and extension tests
- model registry and verification utilities
- model card, architecture notes, security notes, roadmap, and release documentation

## Tech Stack

| Area | Technology |
| --- | --- |
| ML runtime | TensorFlow / Keras |
| Image processing | OpenCV, Pillow, NumPy |
| Inference API | Python, FastAPI, Uvicorn, Pydantic |
| Browser client | Chrome Manifest V3, JavaScript |
| Testing | pytest, Node-based extension tests |
| Model management | JSON model registry + verification scripts |

## Browser Extension

The extension is built with **Manifest V3** and supports two main workflows.

### Manual image analysis

1. Right-click a webpage image.
2. Choose the extension analysis action.
3. The selected image is sent to the configured API.
4. Detected faces are analyzed individually.
5. Results are shown in the popup/results UI.

### Visible-page scanning

1. Open the extension popup.
2. Choose `Scan visible images`.
3. Eligible visible images are queued and deduplicated.
4. Predictions are cached to avoid unnecessary repeated inference.
5. Per-face overlays are rendered on the page.
6. Scanning can be stopped, cleared, repeated, or disabled for the current site.

By default, the extension talks to the local service at:

```text
http://127.0.0.1:8000
```

## ML Baseline

The currently registered baseline is `legacy-cnn-v1`, a TensorFlow/Keras convolutional neural network.

### Verified architecture

```text
256x256 RGB face
      |
Conv2D 32 + ReLU
MaxPool
      |
Conv2D 64 + ReLU
MaxPool
      |
Conv2D 128 + ReLU
MaxPool
      |
Flatten
Dense 128 + ReLU
Dropout 0.5
Dense 1 + Sigmoid
```

The saved model contains approximately **14.8 million trainable parameters**.

The model outputs a `real_score`; the runtime derives:

```text
fake_score = 1 - real_score
```

The current provisional interpretation is:

```text
fake_score > 0.60       -> Likely Fake
0.40 to 0.60            -> Uncertain
fake_score < 0.40        -> Likely Real
```

These thresholds are application heuristics, not calibrated probabilities.

## Reproduced Baseline Evaluation

The repository includes a reproduced evaluation on the balanced local test split used by the legacy model audit:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9397 |
| Balanced accuracy | 0.9397 |
| Fake precision | 0.9300 |
| Fake recall | 0.9510 |
| Fake F1 | 0.9404 |
| Real precision | 0.9499 |
| Real recall | 0.9284 |
| Real F1 | 0.9390 |
| ROC-AUC using fake score | 0.9847 |

These results apply only to the reproduced local dataset split. **No external holdout validation has been completed**, so the numbers should not be interpreted as real-world deepfake detection performance.

See `MODEL_CARD.md` for the full model audit, known validation flaw, dataset details, and prohibited interpretations.

## Why the Model Is Replaceable

The browser extension and API should not have to change every time a better model is trained.

Models are therefore selected through `models/registry.json` and loaded behind the `model_runtime` provider boundary. A replacement model can keep the same input/output contract while the rest of the application remains stable.

This separates:

```text
Browser UX
API contract
Face pipeline
Model implementation
```

and makes the project useful even while the ML baseline is being improved.

## Quick Start

Create a virtual environment and install API dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-api.txt
```

Place the current legacy model at:

```text
models/artifacts/deepfake_detector_93acc.h5
```

or configure an explicit local path:

```powershell
$env:DEEPFAKE_MODEL_PATH="model/deepfake_detector_93acc.h5"
python scripts/verify_model.py --model-id legacy-cnn-v1
```

Start the API:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Install the Extension

1. Open Chrome/Chromium extension management.
2. Enable **Developer mode**.
3. Choose **Load unpacked**.
4. Select the `extension/` directory.
5. Keep the local API running unless extension mock mode is enabled.

## Test Commands

```powershell
python -m compileall api face_pipeline model_runtime scripts
pytest api/tests face_pipeline/test_pipeline.py
npm run test:extension
node --check extension/content-script.js
python scripts/scan_extension_safety.py extension
```

Actual-model tests are skipped when the model artifact is unavailable. Ordinary CI does not require the private/local model file, a dataset, or a GPU.

## Privacy Behavior

With default local settings:

- only user-selected images or images included in a user-controlled scan are analyzed
- images are sent to the local API
- raw image bytes and face crops are not stored in the extension cache

If a remote API origin is configured, eligible images are transmitted to that configured origin instead.

## Repository Structure

```text
api/                FastAPI inference service + tests
extension/          Chrome Manifest V3 extension
face_pipeline/      Face detection and crop utilities
model_runtime/      Replaceable model-provider boundary
models/             Model registry and artifact instructions
scripts/            Model download/verification and safety utilities
evaluation/         Legacy model evaluation tooling
training/           Experimental replacement-model training utilities
reports/            Audit, phase, and release-preparation reports
```

Additional documentation:

- `ARCHITECTURE.md` — component boundaries and data flow
- `MODEL_CARD.md` — model facts, evaluation, limitations, intended use
- `SECURITY.md` — security and privacy considerations
- `ROADMAP.md` — replacement-model and validation work

## Current Limitations

- the active model is an experimental baseline
- real-world webpage predictions remain unreliable
- no unrelated external dataset validation has been completed
- model scores are uncalibrated
- face detection is strongest on visible frontal faces
- compression, screenshots, generator shift, occlusion, and non-frontal faces are not comprehensively validated
- the local API must be running for normal non-mock extension use

## Next ML Work

The next model-focused phase is designed around improving evaluation quality rather than simply increasing model complexity:

- collect multiple real/fake datasets
- use identity-aware or source-aware train/validation/test splits
- train a lighter replacement baseline such as MobileNetV3Large
- evaluate on external holdout sources
- test robustness under compression, noise, and resizing
- calibrate uncertainty thresholds
- replace the legacy provider without changing the browser/API contract

## Interview Summary

> I built an end-to-end human-face deepfake detection prototype consisting of a Chrome Manifest V3 extension, a local FastAPI inference service, a face-detection/cropping pipeline, and a replaceable TensorFlow model runtime. The extension can scan visible webpage images, deduplicate and queue inference requests, and render per-face results while keeping the ML model isolated behind a stable provider interface so it can be replaced after better external validation.
