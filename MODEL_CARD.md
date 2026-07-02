# Model Card: legacy-cnn-v1

## Status

The current model is an experimental placeholder and is marked replaceable. It should not be treated as proof of authenticity.

## Verified Facts

- Model ID: `legacy-cnn-v1`.
- Source artifact: `model/deepfake_detector_93acc.h5` when available locally.
- Framework: TensorFlow/Keras.
- File size observed in audit: 178,116,384 bytes.
- Input: RGB face image resized to 256x256, float32, normalized by dividing by 255.0.
- Output: one sigmoid score interpreted as `real_score`.
- Class mapping: `fake = 0`, `real = 1`.
- Fake score formula: `fake_score = 1.0 - real_score`.
- Provisional thresholds: likely fake below 0.40, uncertain from 0.40 through 0.60, likely real above 0.60.

## Architecture

Verified saved architecture:

- Conv2D 32, ReLU.
- MaxPooling2D.
- Conv2D 64, ReLU.
- MaxPooling2D.
- Conv2D 128, ReLU.
- MaxPooling2D.
- Flatten.
- Dense 128, ReLU.
- Dropout 0.5.
- Dense 1, sigmoid.

Parameter count: 14,839,105 trainable parameters.

## Training Dataset

The reproduced audit used the local Real and Fake Faces dataset split under `data/raw/real_vs_fake/real-vs-fake` with CSV metadata in `data/raw/train.csv`, `data/raw/valid.csv`, and `data/raw/test.csv`.

The available split contains:

- train: 50,000 fake and 50,000 real
- valid: 10,000 fake and 10,000 real
- test: 10,000 fake and 10,000 real

## Reproduced Test Metrics

On the prepared local test split:

- Accuracy: 0.939700.
- Balanced accuracy: 0.939700.
- Precision fake: 0.929982.
- Recall fake: 0.951000.
- F1 fake: 0.940374.
- Precision real: 0.949867.
- Recall real: 0.928400.
- F1 real: 0.939011.
- ROC-AUC using fake score: 0.984720.
- False-positive rate: 0.071600.
- False-negative rate: 0.049000.

False positive means a real face was incorrectly labelled fake. False negative means a fake face was incorrectly labelled real.

## Known Validation Flaw

The original notebook-style training export used `validation_split=0.2` against an ordered training CSV, producing an imbalanced internal validation subset. The reproduced metrics above come from the separate balanced `test.csv`, not from that flawed internal validation split.

## External Validation

No external dataset validation has been completed. The model has not been proven on unrelated generators, webpage crops, screenshots, social-media compression, videos, occluded faces, non-frontal faces, or identity-aware splits.

## Intended Use

- Local experimental MVP testing.
- Architecture validation for face detection, API, extension scanning, and overlays.
- Baseline comparison while developing a replacement model.

## Prohibited Interpretations

- Do not present a prediction as proof that an image is real or fake.
- Do not use scores as calibrated confidence.
- Do not use this model for moderation, enforcement, identity claims, or high-stakes decisions.
- Do not tune thresholds to hide known model failures.

## Replacement Status

The model is loaded through `model_runtime` and selected through `models/registry.json`. It can be replaced without changing the API or extension contract.
