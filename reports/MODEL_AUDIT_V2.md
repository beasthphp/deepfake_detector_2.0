# Human Face Deepfake Model Audit V2

Date: 2026-07-02

Scope: existing human-face deepfake classifier only. This audit does not train a new model, build face detection, build an API, or build a browser extension.

## Verification Run

Files inspected:

- `model/deepfake_detector_93acc.h5`
- `model/umr.py`
- `previous_works/deepfake-detector-main/app.py`
- `data/raw/train.csv`
- `data/raw/valid.csv`
- `data/raw/test.csv`
- `data/raw/real_vs_fake/real-vs-fake`

Important path note:

- The real model is `model/deepfake_detector_93acc.h5`.
- The old app-local file `previous_works/deepfake-detector-main/deepfake_detector_93acc.h5` is still a 134-byte Git LFS pointer.
- `app.py` loads `deepfake_detector_93acc.h5` from the current working directory, so running the app from `previous_works/deepfake-detector-main` would still try to load the pointer unless the real model is copied there or the path is changed later.

Environment:

- Created workspace venv: `.venv-model-audit`
- Python: 3.11.9
- TensorFlow: 2.20.0
- Keras: 3.10.0
- `pip check`: no broken requirements found
- Pinned file created: `requirements-model-audit.txt`

Compatibility finding:

- TensorFlow/Keras 2.15 failed to load the model because the HDF5 config used Keras 3-style `InputLayer(batch_shape=...)`.
- HDF5 metadata reports `keras_version = 3.10.0`.
- TensorFlow 2.20.0 plus Keras 3.10.0 loaded the model successfully.

Generated evaluation files:

- `reports/existing_model/metrics.json`
- `reports/existing_model/classification_report.txt`
- `reports/existing_model/predictions.csv`
- `reports/existing_model/confusion_matrix.png`
- `reports/existing_model/roc_curve.png`
- `reports/existing_model/evaluation_summary.md`
- Extra audit artifacts: `model_summary.txt`, `model_inspection.json`, `score_histogram.png`, `score_distributions.json`, `sample_predictions.json`, `validation_predictions.csv`, `validation_score_histogram.png`, `validation_uncertainty_analysis.json`

## A. Verified Architecture

The saved model is a real HDF5/Keras model, not a Git LFS pointer and not corrupted. It loaded successfully with compile metadata.

Model file:

- Path: `model/deepfake_detector_93acc.h5`
- Size: 178,116,384 bytes
- Input shape: `(None, 256, 256, 3)`
- Output shape: `(None, 1)`
- Output activation: sigmoid
- Optimizer metadata: Adam
- Loss metadata: binary crossentropy
- Stored metric metadata: compile/loss metadata was present; the original `accuracy` string is visible in HDF5 training config, but Keras 3 reports loaded compile metrics lazily.

Architecture:

| Layer | Output shape | Activation | Parameters |
| --- | --- | --- | ---: |
| InputLayer | `(None, 256, 256, 3)` | none | 0 |
| Conv2D, 32 filters, 3x3 valid | `(None, 254, 254, 32)` | relu | 896 |
| MaxPooling2D, 2x2 | `(None, 127, 127, 32)` | none | 0 |
| Conv2D, 64 filters, 3x3 valid | `(None, 125, 125, 64)` | relu | 18,496 |
| MaxPooling2D, 2x2 | `(None, 62, 62, 64)` | none | 0 |
| Conv2D, 128 filters, 3x3 valid | `(None, 60, 60, 128)` | relu | 73,856 |
| MaxPooling2D, 2x2 | `(None, 30, 30, 128)` | none | 0 |
| Flatten | `(None, 115200)` | none | 0 |
| Dense, 128 units | `(None, 128)` | relu | 14,745,728 |
| Dropout, 0.5 | `(None, 128)` | none | 0 |
| Dense, 1 unit | `(None, 1)` | sigmoid | 129 |

Parameter counts:

- Model parameters from `model.count_params()`: 14,839,105
- Trainable parameters: 14,839,105
- Non-trainable parameters: 0
- Keras summary total including optimizer params: 14,839,107

The saved architecture matches the CNN defined in `umr.py`.

Flatten analysis:

- The final convolution output is `30 x 30 x 128 = 115,200` values.
- The next dense layer has 128 units.
- Dense parameters from this single block: `(115,200 x 128) + 128 = 14,745,728`.
- That is about 99.37% of all model parameters.
- The dense kernel alone takes 58,982,400 bytes as float32 weights.

## B. Verified Preprocessing

Training CNN path in `umr.py`:

- Loader: `ImageDataGenerator`
- Rescale: `1./255`
- Image source: `flow_from_dataframe`
- `x_col`: `path`
- `y_col`: `label_str`
- `class_mode`: `binary`
- `target_size`: `(256, 256)`
- `batch_size`: 32
- `color_mode`: default `rgb`
- Interpolation: default `nearest`
- Channel order: RGB, channels-last
- Batch shape: `(batch, 256, 256, 3)`

Evaluation path:

- Uses `ImageDataGenerator(rescale=1./255)`
- Uses `flow_from_dataframe`
- Explicit `classes=["fake", "real"]`
- Explicit `color_mode="rgb"`
- Explicit `interpolation="nearest"`
- Streams batches from disk and does not load all 20,000 test images into RAM.

App path in `app.py`:

- Opens uploaded image with PIL.
- Converts to RGB.
- Resizes to `(256, 256)`.
- Converts to array.
- Adds batch dimension.
- Divides by 255.0.

Preprocessing mismatch:

- Keras training/evaluation uses `interpolation="nearest"` when resizing.
- The app uses `PIL.Image.resize((256, 256))` with no resample argument. In Pillow 10.4.0 this resolves to bicubic for RGB images.
- The current dataset images are already 256x256, so interpolation does not affect the reproduced dataset metrics.
- For arbitrary uploaded images, interpolation could slightly change model scores. The app should later use the same preprocessing function as evaluation/training.

Other preprocessing notes:

- An earlier experimental `cv2.imread` section in `umr.py` uses OpenCV BGR, but this is not the final CNN training path.
- The final CNN training path uses the Keras generator, which reads RGB.

## C. Verified Class Mapping

Verified from CSV metadata:

- `label = 0`, `label_str = fake`
- `label = 1`, `label_str = real`

Verified from the actual Keras generator:

- `class_indices = {"fake": 0, "real": 1}`

Verified output interpretation:

- Model output is `real_score`.
- `real_score` near 0 means fake.
- `real_score` near 1 means real.
- `fake_score = 1.0 - real_score`.
- Threshold used for reproduced metrics: `real_score >= 0.5` predicts real; otherwise fake.

There is no class reversal in the evaluation path.

## D. Reproduced Test Metrics

Evaluation split:

- `data/raw/test.csv`
- 20,000 images
- 10,000 fake
- 10,000 real

Metrics on this exact test split:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.939700 |
| Balanced accuracy | 0.939700 |
| Precision, fake | 0.929982 |
| Recall, fake | 0.951000 |
| F1-score, fake | 0.940374 |
| Precision, real | 0.949867 |
| Recall, real | 0.928400 |
| F1-score, real | 0.939011 |
| ROC-AUC using fake score | 0.984720 |
| False-positive rate | 0.071600 |
| False-negative rate | 0.049000 |

Definitions:

- False positive: a real face incorrectly labelled fake.
- False negative: a fake face incorrectly labelled real.

Confusion matrix with labels `[fake, real]`:

| True / Predicted | fake | real |
| --- | ---: | ---: |
| fake | 9,510 | 490 |
| real | 716 | 9,284 |

Claimed `93acc` check:

- The reproduced test accuracy is 93.97%.
- This approximately reproduces the filename claim on this exact prepared test split.
- This must not be described as 93% accurate on all real-world deepfakes.

Deterministic individual predictions, seed 69:

| Path | True label | Raw real_score | Fake score | Predicted | Correct |
| --- | --- | ---: | ---: | --- | --- |
| `test/real/30872.jpg` | real | 0.999211 | 0.000789 | real | true |
| `test/real/27074.jpg` | real | 0.994382 | 0.005618 | real | true |
| `test/real/17420.jpg` | real | 0.998855 | 0.001145 | real | true |
| `test/real/45130.jpg` | real | 0.593680 | 0.406320 | real | true |
| `test/real/48312.jpg` | real | 0.987238 | 0.012762 | real | true |
| `test/fake/6MJFQXMTQ1.jpg` | fake | 0.161938 | 0.838062 | fake | true |
| `test/fake/KDVBU1XEB9.jpg` | fake | 0.000045 | 0.999955 | fake | true |
| `test/fake/ADJWQO1GRK.jpg` | fake | 0.622778 | 0.377222 | real | false |
| `test/fake/YA9YTPHW20.jpg` | fake | 0.000513 | 0.999487 | fake | true |
| `test/fake/YQN0ECMOSQ.jpg` | fake | 0.098759 | 0.901241 | fake | true |

Score-distribution summary on test:

- Correct real median `real_score`: 0.979189
- Incorrect real median `real_score`: 0.253059
- Correct fake median `real_score`: 0.000257
- Incorrect fake median `real_score`: 0.743542

Validation-based uncertainty probe:

- A `0.45 to 0.55` real-score range marked 303/20,000 validation images uncertain and gave 0.945017 accuracy on the remaining certain images.
- A `0.40 to 0.60` range marked 638/20,000 validation images uncertain and gave 0.951761 accuracy on the remaining certain images.
- A `0.35 to 0.65` range marked 962/20,000 validation images uncertain and gave 0.956718 accuracy on the remaining certain images.

Provisional recommendation only:

- `0.40 to 0.60` is a reasonable provisional uncertainty range to investigate next because it catches about 3.19% of validation images and improves accuracy on the remaining certain set.
- Do not adopt this as a product threshold yet. It needs calibration and external validation.

## E. Model-Size Analysis

The file is about 178 MB primarily because of the Flatten-to-Dense block and saved Adam optimizer slots.

Measured HDF5 dataset bytes:

- `model_weights`: 59,356,420 bytes
- `optimizer_weights`: 118,712,852 bytes

Largest datasets:

- Dense kernel weights: 58,982,400 bytes
- Adam momentum for dense kernel: 58,982,400 bytes
- Adam velocity for dense kernel: 58,982,400 bytes

Interpretation:

- The model's trainable weights are about 56.6 MB.
- Adam stores two additional slot tensors, momentum and velocity, for the large dense kernel.
- Those optimizer tensors make the `.h5` roughly three times larger than inference weights alone.
- Saving an inference-only model later would substantially reduce the deployed artifact size, but the original `.h5` should not be overwritten.

## F. Generalization Limitations

This evaluation is meaningful but limited.

- The test split comes from the same overall 140K Real and Fake Faces dataset family as training.
- It does not prove performance on unrelated generators, social-media compression, screenshots, partially occluded faces, face crops from web pages, videos, or non-FFHQ-like distributions.
- The model could still be using generator-specific or dataset-specific artifacts.
- The current model classifies already prepared face images. It is not a face detector.
- Future face detection/cropping must remain a separate pipeline stage before deepfake classification.

Important `umr.py` training-pipeline issue:

- The CNN code uses `train.csv` plus `ImageDataGenerator(validation_split=0.2)`.
- With the current ordered `train.csv`, the generated validation subset contains 20,000 real images and 0 fake images.
- The generated training subset contains 50,000 fake images and 30,000 real images.
- Therefore, original validation accuracy from that code would be misleading and not class-balanced.
- The reproduced 93.97% result above is from the separate balanced `test.csv`, not from that flawed internal validation split.

Other `umr.py` notes:

- The file is a Colab export and includes shell commands such as `!kaggle`, so it is not directly runnable as normal Python without cleanup.
- It contains broad `except Exception as e: pass` in the FFT feature extraction path, which can silently hide data failures.
- The RandomForest block uses `accuracy_score` before the metrics import appears later in the notebook export.
- The MLP comment says scaled data is critical, but the shown MLP fit uses `X_train` directly rather than a scaler pipeline.
- These classical experiments should remain experimental baselines only.

## G. Existing Model Verdict

Verdict: usable but needs improvement.

Why:

- It loads successfully when using the correct Keras 3.10-compatible environment.
- It approximately reproduces the claimed 93% result on the prepared balanced test split.
- It has strong test ROC-AUC on this dataset.

Why it still needs improvement:

- The original training validation split in `umr.py` is flawed.
- The model is large for such a small CNN because `Flatten` feeds a huge dense layer.
- The app currently points at an old local LFS pointer unless the working directory/model path is corrected later.
- App resize interpolation does not exactly match the training generator for non-256x256 uploads.
- There is no external cross-dataset evaluation.
- Confidence is not calibrated for real-world deployment.

## H. Recommended Next Model Experiment

| Option | Strength | Weakness | Fit for this project |
| --- | --- | --- | --- |
| Existing CNN | Reproduced 93.97% test accuracy on current split; simple baseline. | 14.8M params, 178 MB training artifact, flawed original validation split, likely overfit risk from Flatten. | Keep as baseline, not final direction. |
| MobileNetV3Small | Very small and fast; best browser-side candidate. | May underfit subtle deepfake artifacts compared with larger transfer models. | Good later if browser inference dominates. |
| MobileNetV3Large | Strong size/accuracy/speed balance; much more deployable than current HDF5; suitable for transfer learning. | Slightly heavier than Small; may still need careful calibration. | Best next experiment. |
| EfficientNetB0 | Strong general transfer-learning baseline with good parameter efficiency. | Often slower/heavier than MobileNetV3 on CPU/browser paths. | Good comparison after MobileNetV3Large. |
| Xception | Strong image-forensics-style backbone. | Much larger and slower; poor browser-side fit. | Server-side research comparison only. |

Recommended next model experiment: MobileNetV3Large.

Do not train it yet. First clean the training pipeline so comparisons are fair and reproducible.

## Exact Next Action

Refactor only the training/evaluation pipeline structure next: make it use explicit `train.csv`, `valid.csv`, and `test.csv` splits, a shared preprocessing function, and reproducible metric reporting, then rerun the existing CNN training once before starting the MobileNetV3Large experiment.
