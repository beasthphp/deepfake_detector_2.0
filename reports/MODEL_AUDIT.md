# Human Face Deepfake Model Audit

Date: 2026-07-02

Scope: human-face deepfake classification only. This report does not propose a general AI-image detector, browser extension, or API implementation.

## Verification Summary

Safe checks run:

- Repository inventory with `rg --files` and targeted PowerShell listings.
- `app.py` syntax compile: passed.
- Python version: `Python 3.13.9`.
- Dependency import check: `streamlit`, `tensorflow`, `keras`, `numpy`, `PIL`, `huggingface_hub`, and `sklearn` were not installed in the local environment.
- TensorFlow import check: failed with `ModuleNotFoundError: No module named 'tensorflow'`.
- Local saved model check: `previous_works/deepfake-detector-main/deepfake_detector_93acc.h5` is a 134-byte Git LFS pointer, not a loadable HDF5/Keras model.
- CSV split audit: completed.
- Image path, class balance, image header, dimension, JPEG quantization-table, and cross-split exact duplicate hash checks: completed.
- A first all-in-one image scan timed out and is not counted as a completed check; it was replaced by smaller completed checks.

## A. Existing System Summary

The local project contains a Streamlit-style inference app and a prepared face dataset tree. The app accepts an uploaded image, converts it to RGB, resizes it to 256x256, normalizes pixel values by dividing by 255.0, runs `model.predict`, reads `prediction[0][0]`, and applies a 0.5 threshold.

Visible project files:

- `previous_works/deepfake-detector-main/app.py`
- `previous_works/deepfake-detector-main/deepfake_detector_93acc.h5`
- `previous_works/deepfake-detector-main/requirements.txt`
- `previous_works/deepfake-detector-main/README.md`
- `previous_works/deepfake-detector-main/.gitattributes`
- `data/raw/train.csv`
- `data/raw/valid.csv`
- `data/raw/test.csv`
- `data/raw/real_vs_fake/real-vs-fake/{train,valid,test}/{fake,real}`

No local training script, notebook, `umr.py`, FFT experiment script, classical ML script, saved scikit-learn model, or evaluation script was found in this checkout.

The app tries to load `deepfake_detector_93acc.h5` locally first. Because the local file exists but is only a Git LFS pointer, the app will attempt to load the pointer file instead of downloading from Hugging Face. Also, the local environment does not currently have TensorFlow installed, so model loading could not be tested here.

## B. Current Model Architecture

Verified from local files:

- Model file name: `deepfake_detector_93acc.h5`
- Local file status: Git LFS pointer, not actual HDF5 model data.
- LFS pointer target size: 178,116,384 bytes.
- Inference input preprocessing in `app.py`: RGB image resized to 256x256, expanded to batch shape, divided by 255.0.
- Inferred input shape for inference: `(None, 256, 256, 3)`.
- Inferred output usage: one scalar score, interpreted as probability-like score for class `real`.
- Threshold: score `< 0.5` means fake; score `>= 0.5` means real.

Not verifiable from this checkout:

- Convolution block structure.
- Whether the model uses `Flatten()`.
- Dense layer sizes.
- Exact output activation.
- Trainable parameter count.
- Optimizer, loss, training history, or validation/test metrics.

Because the real `.h5` model is absent and TensorFlow is not installed locally, the model summary and parameter count could not be calculated. The 178 MB LFS target size suggests a potentially large model artifact, but it is not a valid substitute for a parameter count.

Architecture audit answer:

- A custom CNN can be a useful first baseline for 256x256 face deepfake classification, but it is usually weaker than transfer learning when data sources differ from real-world deployment images.
- If the model uses `Flatten()` after large convolution maps, it can create an unnecessarily large dense layer and raise overfitting risk. This specific model cannot be confirmed to use `Flatten()` from the available files.
- The project has no reproducible evidence here that the current architecture generalizes beyond the prepared dataset.
- 256x256 is a reasonable face-crop size for server-side inference and training. For future browser-side inference, 224x224 or 192x192 may be worth testing with lightweight models.

## C. Verified Class Mapping

The class mapping is verified from `data/raw/*.csv` and from the inference app:

- `label = 0`, `label_str = fake`
- `label = 1`, `label_str = real`

The app uses the same interpretation:

- `score < 0.5`: fake
- `score >= 0.5`: real
- Fake confidence displayed as `(1 - score) * 100`
- Real confidence displayed as `score * 100`

This is internally consistent if the model output is a sigmoid-like probability for class `real`. The actual model output activation could not be verified because the real model artifact is missing.

If training used Keras `flow_from_directory`, the likely alphabetical class index would be `fake = 0`, `real = 1`, but no generator code or `class_indices` output is present locally. The verified mapping in this audit comes from the CSV files and app logic, not from training code.

## D. Critical Problems

1. The actual saved model is missing from the checkout.

   `deepfake_detector_93acc.h5` is a Git LFS pointer:

   - local size: 134 bytes
   - pointer target size: 178,116,384 bytes
   - not an HDF5/Keras file

   This prevents model summary, parameter count, output shape verification, and prediction tests.

2. The local dependency environment is not usable for inference.

   `requirements.txt` lists unpinned packages only:

   - `streamlit`
   - `tensorflow-cpu`
   - `numpy`
   - `Pillow`
   - `huggingface_hub`

   None were installed in the current local Python environment.

3. The `93acc` filename is not supported by a reproducible local test result.

   No training history, evaluation script, confusion matrix, test predictions, or metrics file was found. The report cannot treat this model as "93% accurate on all deepfakes." At most, the filename claims some unknown accuracy from an unknown split or run.

4. Training code is absent.

   There is no local evidence for the CNN architecture, augmentation, generator configuration, validation split, callbacks, checkpointing, or class index generation.

5. Evaluation is absent.

   The project currently does not calculate or store balanced accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, false-positive rate, or false-negative rate in the visible files.

6. The app forces a binary verdict.

   The intended pipeline needs `Likely Fake`, `Uncertain`, and `Likely Real`. The current app has no uncertainty band and presents every score as either fake or real.

7. Confidence is not calibrated.

   The percentage calculation is mathematically consistent with the app's class mapping, but it should be treated as model score, not proof or calibrated real-world confidence.

8. The app analyzes any uploaded image as though it were a face.

   This is acceptable for an early app demo only if users supply cropped human faces. The final pipeline should keep face detection and cropping separate from deepfake classification.

9. Dataset-source learning remains a risk.

   The prepared dataset combines real FFHQ images and fake generated-face images. Even though local checks found consistent dimensions and JPEG quantization tables, the model could still learn dataset-specific artifacts, generator-specific artifacts, alignment conventions, or source-specific face distributions instead of robust deepfake evidence.

## E. Important Improvements

- Restore the actual Git LFS model file or download the exact Hugging Face artifact, then run a read-only `model.summary()` and parameter count.
- Pin the runtime environment with exact Python and package versions.
- Add a reproducible training script or notebook export.
- Add a reproducible evaluation script for the existing test split.
- Store evaluation outputs: accuracy, balanced accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, false-positive rate, and false-negative rate.
- Use one shared preprocessing function for training and inference.
- Add an uncertainty band around the threshold, for example `0.4 <= score <= 0.6`, then tune that band using validation and calibration data.
- Use cautious UI language: "likely fake", "uncertain", and "likely real", not absolute proof.
- Add face detection/cropping as a separate stage later, before classification.
- Replace broad exception handling with logged, specific failure paths. The code does not contain `except Exception: pass`, but it does contain broad `except Exception as e` blocks.
- Fix model-loading behavior so a Git LFS pointer or corrupt local model does not prevent fallback to the remote model.

## F. Optional Improvements

- Add model calibration checks, such as reliability curves and expected calibration error.
- Add Grad-CAM or saliency diagnostics for debugging, while avoiding overclaiming explainability.
- Add TFLite or ONNX export only after the model is reproducible and evaluated.
- Add face alignment and crop-margin experiments after the base classifier is validated.
- Add metadata reports for dataset provenance and split creation.
- Preserve an experimental FFT/classical ML baseline if it exists outside this checkout.

## G. Reusable Components

Useful pieces to preserve:

- The prepared `data/raw/real_vs_fake/real-vs-fake` split structure.
- The CSV metadata with explicit `label` and `label_str` columns.
- The app's basic image ingestion flow.
- RGB conversion, 256x256 resizing, and `/255.0` normalization, if confirmed to match training.
- The binary score interpretation, if confirmed by the real model summary and training code.
- The Hugging Face fallback idea, after local pointer handling is fixed.

Pieces that are incomplete or need cleanup:

- `README.md` is only a title and does not document reproduction steps.
- `requirements.txt` is unpinned.
- `deepfake_detector_93acc.h5` is unresolved Git LFS content.
- `.devcontainer/devcontainer.json` installs unpinned requirements and launches Streamlit, but it does not solve the missing LFS model in this checkout.

## H. Recommended Model Direction

No accuracy numbers can be promised from the local evidence. The comparison below is a project-direction recommendation for a human-face deepfake classifier, not a reported benchmark.

| Option | Expected accuracy/generalization | Model size | Training cost | CPU inference speed | FastAPI suitability | Browser-side suitability |
| --- | --- | --- | --- | --- | --- | --- |
| Existing custom CNN | Unknown. Could be a useful baseline, but no architecture or metrics are reproducible here. Higher overfitting risk if it uses large dense layers. | Unknown. The LFS artifact target is 178 MB, which is large for deployment. | Low to medium. | Unknown. | Possible if restored and fast enough. | Questionable until size and latency are known. |
| MobileNetV3 | Often strong for lightweight transfer learning, especially when deployment matters. May trail larger models on maximum accuracy. | Small. Good TFLite path. | Low. | Fast. | Good. | Best of these options. |
| EfficientNetB0 | Strong transfer-learning baseline with good accuracy/size balance. Often better accuracy than very small mobile models. | Small to medium. | Low to medium. | Moderate. | Very good. | Possible, but heavier than MobileNetV3. |
| Xception | Strong feature extractor for image forensics-style tasks, but heavier and easier to overfit without careful evaluation. | Large. | Higher. | Slower. | Good on a server with enough CPU/GPU budget. | Poor fit for browser-side inference. |

Recommended next baseline: MobileNetV3.

Reason: this project has a future FastAPI and possible browser-extension path. MobileNetV3 gives the best deployment tradeoff while still benefiting from pretrained features. EfficientNetB0 should be the next comparison if MobileNetV3 accuracy is not enough. Xception is better kept as a heavier server-side experiment, not the first deployable baseline.

## I. Proposed Step-by-Step Roadmap

1. Audit the existing model.
2. Clean and refactor the training pipeline.
3. Reproduce the existing CNN result.
4. Add reliable evaluation and test-set separation.
5. Train a transfer-learning baseline.
6. Compare the custom CNN with the improved model.
7. Add face detection and cropping.
8. Create a FastAPI inference service.
9. Build the browser extension.
10. Optimize performance and confidence thresholds.

## Dataset Pipeline Audit

CSV metadata and local image tree:

| Split | Fake | Real | Total |
| --- | ---: | ---: | ---: |
| train | 50,000 | 50,000 | 100,000 |
| valid | 10,000 | 10,000 | 20,000 |
| test | 10,000 | 10,000 | 20,000 |

Verified:

- All CSV paths checked in the audit existed locally.
- No duplicate `id` values across splits.
- No duplicate `original_path` values across splits.
- No duplicate relative `path` values across splits.
- Exact file-hash duplicate check across train/valid/test found 0 cross-split duplicate hashes.
- All 140,000 local images were readable as JPEGs by the header parser.
- All 140,000 local images reported dimensions of 256x256.
- JPEG quantization-table fingerprint was the same across real and fake classes in all splits.
- File-size medians were close but not identical:
  - train fake median: 28,012 bytes
  - train real median: 28,721 bytes
  - valid fake median: 27,896 bytes
  - valid real median: 28,714 bytes
  - test fake median: 27,941 bytes
  - test real median: 28,755 bytes

Not found in local code:

- No `train_test_split` call.
- No `stratify=y` usage.
- No TensorFlow generator configuration.
- No `validation_split` usage.
- No dataset download script.
- No data preparation script.
- No silent `except Exception: pass` block.

Interpretation:

- The available dataset is already split and class-balanced.
- There is no local evidence of random splitting leakage in the visible code.
- Identity-aware splitting cannot be verified from the current metadata. The `id` column appears to be an image/source identifier, not a person identity. To do identity-aware splitting, the project would need identity labels or a face-embedding clustering pass.
- Compression and image-size artifacts are less concerning than expected because all images are 256x256 JPEGs with the same quantization-table fingerprint, but source-specific and generator-specific artifacts are still possible.

## Evaluation Quality Audit

Currently visible project evidence:

- Accuracy: not reproducibly calculated in local files.
- Balanced accuracy: not found.
- Precision: not found.
- Recall: not found.
- F1-score: not found.
- ROC-AUC: not found.
- Confusion matrix: not found.
- False-positive rate: not found.
- False-negative rate: not found.

The `93acc` filename is not enough evidence. It does not say whether the number came from training, validation, test, or leaked data. It also does not show class-wise behavior. For the intended product, false positives and false negatives matter because a face could be incorrectly accused as fake or incorrectly trusted as real.

The current app's confidence percentage is a display transformation of a single model score. It should be described as a score, not as proof.

## Classical ML and FFT Experiments Audit

No FFT, logistic regression, MLP, scikit-learn, or classical ML experiment files were found in this checkout.

Based on the described feature set:

- Mean frequency magnitude, frequency standard deviation, and mean pixel value are not sufficient for real-world face deepfake detection.
- They can be retained as an experimental baseline.
- They may be useful later as auxiliary features in a CNN-plus-frequency ensemble, but only after proper validation.
- Logistic Regression and MLP baselines should use scaling, such as `StandardScaler`, before fitting.
- Metrics imports and comments could not be audited because the files were not present locally.
- These experiments should not be removed yet if they exist elsewhere, but they should not be treated as production evidence.

## J. Exact Next Step

Recover the actual `deepfake_detector_93acc.h5` model artifact and run a read-only model summary in a pinned TensorFlow environment so the architecture, parameter count, input/output shapes, and reproducible test metrics can be verified.
