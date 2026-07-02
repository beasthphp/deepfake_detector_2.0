# Phase 2 Report

## Repository Inspection

- Model location: `D:\ddp\model\deepfake_detector_93acc.h5`
- Shared preprocessing reference: `training/data_pipeline.py`
- Existing model evaluation code: `evaluation/evaluate_existing_model.py`
- Prepared real/fake sample images: `data/raw/real_vs_fake/real-vs-fake/`
- Phase 2 local samples: `phase2_samples/input/`
- Dependency file: `requirements-model-audit.txt`
- Compatible environment used: Python 3.11.9, TensorFlow 2.20.0, Keras 3.10.0
- Detector packages found locally: OpenCV installed; MediaPipe, MTCNN, RetinaFace, and pytest not installed

## A. Selected Face Detector

Selected detector: **OpenCV Haar cascade `haarcascade_frontalface_default.xml`**.

This is the local OpenCV fallback selected for the Phase 2 Python prototype because `opencv-python==4.10.0.84` is already installed in the working Python 3.11/TensorFlow 2.20 environment. MediaPipe, MTCNN, RetinaFace, and pytest are not installed. OpenCV's modern YuNet API is available, but no YuNet ONNX model file is present locally, so using YuNet would require adding a new model asset.

The longer-term OpenCV choice should be YuNet or MediaPipe. Haar is enough for this phase's plumbing and crop-sensitivity test, but its confidence value is not calibrated and frontal-face coverage is limited.

Decision details are in `reports/PHASE2_FACE_DETECTOR_DECISION.md`.

## B. Pipeline Architecture

```text
Image -> Face detector -> Face cropper -> Preprocessing
-> Deepfake classifier -> Result labels
```

Implemented files:

- `face_pipeline/detector.py`: accepts path, PIL image, or NumPy array; returns sorted `xyxy_exclusive` boxes, detector confidence, face index, and image dimensions.
- `face_pipeline/cropper.py`: square crop with 20% default margin, boundary clamping, and tiny-crop rejection.
- `face_pipeline/classifier.py`: loads `model/deepfake_detector_93acc.h5` once, checks the file is real HDF5, applies RGB/256x256/nearest/float32/255 preprocessing, and converts `real_score` to `fake_score`.
- `face_pipeline/pipeline.py`: CLI and reusable pipeline wrapper.
- `face_pipeline/visualization.py`: annotated output images.
- `face_pipeline/crop_consistency.py`: deterministic original-vs-recropped evaluation.
- `face_pipeline/test_pipeline.py`: unit tests using built-in `unittest`.

Provisional labels:

- `real_score < 0.40`: Likely Fake
- `0.40 <= real_score <= 0.60`: Uncertain
- `real_score > 0.60`: Likely Real

These are validation-derived provisional ranges, not calibrated real-world confidence.

## C. Test Results

Unit tests:

- Command: `python -m unittest face_pipeline.test_pipeline`
- Result: 10 tests ran and passed.

CLI smoke tests:

- `one_clear_real.jpg`: 1 face detected, labelled Likely Real.
- `no_face_geometric.jpg`: 0 faces detected, returned `status: no_face_detected`; the deepfake model was not run.

Deterministic sample set:

| Sample | Expected purpose | Faces detected | Result highlights |
| --- | --- | ---: | --- |
| `one_clear_real.jpg` | one clear face, real | 1 | Likely Real, real_score 0.844038 |
| `one_clear_fake.jpg` | one clear face, fake | 1 | Likely Real, real_score 0.995391 |
| `multiple_faces_real_fake.jpg` | multiple faces | 2 | Both faces detected left-to-right |
| `no_face_geometric.jpg` | no face | 0 | `no_face_detected` |
| `small_face_real.jpg` | small face | 1 | 64x64 detector box, 90x90 crop |
| `edge_face_fake.jpg` | face near boundary | 1 | Crop clamped to left edge |
| `non_square_real.jpg` | non-square image | 1 | Square crop from 520x300 image |
| `already_cropped_256_real.jpg` | prepared 256x256 face | 1 | Likely Real, real_score 0.999282 |

Outputs are saved under `phase2_samples/output/`. The sample run JSON is saved at `reports/phase2/sample_pipeline_results.json`.

## D. Crop-Consistency Results

The crop-consistency run used 25 real and 25 fake test images, selected deterministically with seed `69`. It compared the original prepared image prediction against the prediction after detecting and recropping the same image.

Summary:

- Records evaluated: 50
- Skipped before reaching target: 2
- Average absolute score difference: 0.328043
- Median absolute score difference: 0.057793
- Maximum absolute score difference: 0.999936
- Predicted class changes: 16 / 50
- Uncertainty increased: 27 / 50

By class:

| True class | Count | Avg abs diff | Median abs diff | Class changes | Uncertainty increased |
| --- | ---: | ---: | ---: | ---: | ---: |
| real | 25 | 0.090990 | 0.014527 | 2 | 6 |
| fake | 25 | 0.565096 | 0.865071 | 14 | 21 |

This is the critical result: automatic recropping made many fake examples score as real. Row-level results are in `reports/phase2/crop_consistency.json`, and the focused report is in `reports/phase2/CROP_CONSISTENCY_REPORT.md`.

## E. Performance

Environment:

- Python 3.11.9
- TensorFlow 2.20.0
- Keras 3.10.0
- OpenCV 4.10.0.84
- TensorFlow runtime: CPU only in this environment

Warm crop-consistency averages:

- Face detection: 27.205 ms
- Recropped classification per face: 166.703 ms
- Total processing per image: 371.268 ms

Cold CLI startup is much slower because TensorFlow and the `.h5` model load on first classification. The first real-image CLI run took about 17.5 seconds end to end.

## F. Known Limitations

- Haar is frontal-face oriented and weaker on profile faces.
- Occluded faces and extreme poses may be missed.
- Very small faces are fragile, even though the small synthetic sample was detected.
- Low-resolution or heavily compressed images may produce poor crops.
- Cartoons, animals, and non-human faces are outside this project scope.
- Detector confidence is not a calibrated probability.
- The deepfake classifier was trained/evaluated on prepared face images; automatic crop changes can strongly alter predictions.
- Fake-image recrops are especially unstable in this deterministic test.

## G. Phase 2 Verdict

**Functional but crop pipeline needs adjustment.**

The local face-detection, crop, classifier, CLI, annotation, sample-output, unit-test, and crop-consistency pieces are implemented. However, the model does not work reliably on automatic face crops yet: 16 of 50 crop-consistency examples changed predicted class, including 14 of 25 fake examples.

## H. Exact Next Action

Run a crop-strategy sweep on the same deterministic 50-image consistency set, starting with less aggressive recropping and an option to preserve already-prepared 256x256 face crops, then adopt the crop rule that minimizes fake-to-real flips before API integration.
