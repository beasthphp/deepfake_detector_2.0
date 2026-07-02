# Phase 3 Report

## A. Pre-API Crop Sanity Result

The pre-API sanity check passed. It examined 25 deterministic Phase 2 records and saved row-level output to `reports/phase3/pre_api_crop_sanity.json`.

- Prepared-face bypass fired: 24 / 25
- Final crop equaled the complete original image: 25 / 25
- Expanded contextual crop clamped to the full image: 25 / 25
- Final model-input pixels matched original model-input pixels: 25 / 25
- Conclusion: `expected_because_prepared_faces_occupy_full_frame`

The zero score differences in Phase 2.1 are explained by prepared-face geometry. Most images intentionally used the bypass; the remaining checked image did not bypass because multiple faces were detected, but its expanded context crop still clamped to the full 256x256 frame. No crop implementation bug was found.

## B. API Architecture

```text
HTTP upload
    ->
Validation and image decoding
    ->
Face detection
    ->
Preserve/context crop
    ->
Batched deepfake inference
    ->
Structured JSON
```

The API layer is under `api/`. Face detection, crop strategy, preprocessing, and classification remain in `face_pipeline/`.

## C. Endpoints Implemented

- `GET /health`
- `GET /model-info`
- `POST /predict`

`POST /predict-batch` was not implemented in this first pass.

## D. Validation And Security Controls

- Rejects empty files.
- Enforces 10 MB default upload limit.
- Rejects unsupported declared content types.
- Inspects decoded image format instead of trusting filename extensions.
- Supports JPEG, PNG, and WebP when Pillow decodes them.
- Rejects malformed images.
- Treats decompression-bomb warnings as errors.
- Enforces image pixel and dimension limits.
- Processes uploads in memory.
- Does not save uploads or crops by default.
- Does not return internal stack traces or local paths.
- Uses request IDs in responses.
- Loads TensorFlow model and Haar detector once at startup.
- Verifies the model is real HDF5, not a Git LFS pointer.
- Verifies model input shape `(None, 256, 256, 3)` and output shape `(None, 1)`.
- Uses a configurable concurrency limit, an inference lock, and a request timeout.
- Uses configurable CORS origins instead of unrestricted credentialed CORS.

## E. Functional Test Results

Command:

```powershell
& 'D:\ddp\.venv-model-audit\Scripts\python.exe' -m pytest face_pipeline\test_pipeline.py api\tests -q
```

Result:

- Passed: 34
- Failed: 0
- Warnings: 16 dependency warnings from TensorFlow/matplotlib internals

Covered paths include health, model info, real image inference, fake image inference, multiple faces, no face, tiny image, oversized upload, unsupported text file, corrupted JPEG, extension/content mismatches, face near boundary, too many faces, detector exception, classifier exception, concurrency, and timeout behavior.

## F. Performance Results

Benchmark files:

- `reports/phase3/api_benchmark.json`
- `reports/phase3/API_BENCHMARK.md`

Cold startup:

- Model and detector load: 10690.011 ms
- Memory measurement: unavailable because `psutil` is not installed

Warm request samples:

| Case | HTTP | API status | Faces | Response total ms | Client wall ms |
| --- | ---: | --- | ---: | ---: | ---: |
| one_face | 200 | completed | 1 | 183.949 | 188.944 |
| multiple_faces | 200 | completed | 2 | 341.547 | 347.947 |
| no_face | 200 | no_face_detected | 0 | 10.148 | 16.208 |

Ten sequential warm requests:

- Failed requests: 0 / 10
- Average client wall time: 205.549 ms
- Median client wall time: 205.183 ms
- Average response total time: 199.963 ms

## G. Known Limitations

- Phase 2.1 observed exactly 5% face detection/crop failure on deterministic validation.
- Haar frontal-face detection has known limitations for profile, occluded, tiny, angled, and unusual faces.
- No real-world external dataset validation has been run.
- The classifier was trained on prepared face images.
- Scores are uncalibrated model outputs, not calibrated probabilities or proof.
- This is a CPU-only local prototype.
- The existing model is large and causes a cold-start cost of about 10.69 seconds.
- The API uses a simple inference lock and should not be presented as highly scalable.

## H. Phase 3 Verdict

Ready for controlled extension prototype with documented limitations.

The API starts, validates inputs, avoids translating detector/classifier failures into real labels, returns structured JSON, batches multiple faces inside one request, and has passing tests. It is not a production reliability gate because the Phase 2.1 detector failure rate remains 5%.

## I. Exact Next Action

Build the browser-extension prototype against the local `POST /predict` endpoint, keeping the UI explicit about `no_face_detected`, `Uncertain`, and uncalibrated scores.
