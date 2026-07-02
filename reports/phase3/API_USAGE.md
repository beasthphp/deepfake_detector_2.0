# API Usage

## Startup

Install the API additions in the existing model-audit environment:

```powershell
& 'D:\ddp\.venv-model-audit\Scripts\python.exe' -m pip install -r 'D:\ddp\requirements-api.txt'
& 'D:\ddp\.venv-model-audit\Scripts\python.exe' -m pip check
```

Start the local development server:

```powershell
& 'D:\ddp\.venv-model-audit\Scripts\python.exe' -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Do not expose this development server publicly. FastAPI OpenAPI docs are available at `http://127.0.0.1:8000/docs` while the server is running.

## Endpoints

### `GET /health`

Returns startup status without running inference.

```json
{
  "status": "ok",
  "model_loaded": true,
  "detector_loaded": true,
  "device": "CPU",
  "model_version": "existing-cnn-93acc"
}
```

### `GET /model-info`

Returns model contract, class mapping, detector, crop strategy, provisional thresholds, and known limitations. It does not expose absolute local model paths.

### `POST /predict`

Accepts one uploaded JPEG, PNG, or WebP image as multipart form field `file`.

PowerShell example:

```powershell
Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8000/predict' `
  -Method Post `
  -Form @{ file = Get-Item 'D:\ddp\phase2_samples\input\one_clear_fake.jpg' }
```

curl example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@D:/ddp/phase2_samples/input/one_clear_fake.jpg"
```

Successful face response:

```json
{
  "request_id": "generated-id",
  "status": "completed",
  "image": {"width": 256, "height": 256, "format": "JPEG"},
  "faces_detected": 1,
  "faces": [
    {
      "face_index": 0,
      "bounding_box": {"x1": 50, "y1": 63, "x2": 208, "y2": 221},
      "crop_box": {"x1": 0, "y1": 0, "x2": 256, "y2": 256},
      "face_detection_score": 0.903415,
      "crop_strategy": "preserve_or_context_full_head_l40_r40_t70_b35_square",
      "preserved_original": true,
      "real_score": 0.0019020069,
      "fake_score": 0.9980979931,
      "label": "Likely Fake"
    }
  ],
  "timing_ms": {
    "decode": 0.95,
    "face_detection": 35.789,
    "crop_preprocessing": 0.202,
    "classification": 158.43,
    "serialization": 0.048,
    "total": 195.452
  },
  "warnings": ["Results are probabilistic and are not proof of authenticity."]
}
```

No-face response:

```json
{
  "request_id": "generated-id",
  "status": "no_face_detected",
  "image": {"width": 320, "height": 240, "format": "JPEG"},
  "faces_detected": 0,
  "faces": [],
  "warnings": [
    "No supported human face was detected. The deepfake classifier was not run.",
    "No-face results are never interpreted as Likely Real."
  ]
}
```

## Score Interpretation

- `real_score = float(model_output)`
- `fake_score = 1.0 - real_score`
- `real_score < 0.40`: `Likely Fake`
- `0.40 <= real_score <= 0.60`: `Uncertain`
- `real_score > 0.60`: `Likely Real`

These scores are uncalibrated model outputs, not proof of authenticity.

## Validation And Privacy

- Maximum upload size defaults to 10 MB.
- Maximum image pixel count defaults to 25,000,000.
- Minimum dimensions default to 32x32.
- Maximum dimensions default to 10,000x10,000.
- Uploads are processed in memory and are not saved by default.
- Raw image data and full uploaded filenames are not logged by the API code.
- Metadata is not preserved in decoded images.
- Debug saving is disabled by default through `SAVE_DEBUG_OUTPUTS=false`.

## Configuration

Environment variables:

- `DEEPFAKE_MODEL_PATH`
- `MAX_UPLOAD_BYTES`
- `MAX_IMAGE_PIXELS`
- `MIN_IMAGE_WIDTH`
- `MIN_IMAGE_HEIGHT`
- `MAX_IMAGE_WIDTH`
- `MAX_IMAGE_HEIGHT`
- `MAX_FACES_PER_IMAGE`
- `REQUEST_CONCURRENCY_LIMIT`
- `REQUEST_TIMEOUT_SECONDS`
- `SAVE_DEBUG_OUTPUTS`
- `ALLOWED_ORIGINS` as a comma-separated list

Default CORS origins are `http://localhost` and `http://127.0.0.1`. Configure the future browser-extension origin once the extension ID exists.

## Known Limitations

- Phase 2.1 observed a 5% Haar detection/crop failure rate on deterministic validation.
- Haar is sensitive to non-frontal, occluded, tiny, or unusual faces.
- The classifier was trained on prepared face images.
- The API has not been validated on an external real-world dataset.
- Confidence is uncalibrated.
- CPU-only local inference has a large cold-start cost.
