# Baseline Comparison

Date: 2026-07-02

Scope: prepared 140K Real and Fake Faces test split only. This does not prove real-world deepfake performance.

## Current Status

The original audited CNN has reproducible test metrics. The corrected retraining pipeline has passed the smoke test, but full retraining was not launched because this TensorFlow environment detects CPU only.

Therefore, a fair retrained-vs-original test comparison is pending.

## Metrics

| Metric | Original `deepfake_detector_93acc.h5` | Retrained custom CNN |
| --- | ---: | ---: |
| Test accuracy | 0.939700 | pending full training |
| ROC-AUC | 0.984720 | pending full training |
| Fake precision | 0.929982 | pending full training |
| Fake recall | 0.951000 | pending full training |
| Real precision | 0.949867 | pending full training |
| Real recall | 0.928400 | pending full training |
| False-positive rate | 0.071600 | pending full training |
| False-negative rate | 0.049000 | pending full training |
| Parameter count | 14,839,105 | 14,839,105 expected |
| Model file size | 178,116,384 bytes | pending full checkpoint |

## Smoke Test Result

The corrected pipeline smoke test passed:

- 500 fake and 500 real training images
- 100 fake and 100 real validation images
- 1 epoch
- 0 unreadable images in the smoke subsets
- Checkpoint saved to `D:\ddp\models\experiments\retrained_custom_cnn_smoke_best.keras`
- Validation ROC-AUC: 0.685700

Smoke metrics are not compared against the original model because the smoke subset and one-epoch training run are only a pipeline validation.

## Training Stability

Training stability for the full corrected baseline is pending. The smoke run confirmed that:

- Explicit train and validation CSVs are used.
- Both classes are present in train and validation.
- The class mapping remains `fake = 0`, `real = 1`.
- The shared preprocessing path produces RGB `float32` images at 256x256 in range `[0, 1]`.
- Checkpointing, early-stopping monitoring, ReduceLROnPlateau monitoring, and CSV logging are wired.

## Overfitting Notes

The architecture is unchanged from the original CNN. It still contains the large `Flatten -> Dense(128)` block with 14,745,728 parameters, so overfitting risk remains. Whether the corrected validation split changes the result substantially cannot be answered until full training completes.

## Recommendation

Do not proceed to MobileNetV3Large yet. Run the full corrected custom-CNN baseline in a GPU-capable TensorFlow environment, evaluate it on `test.csv`, then update this comparison with the retrained checkpoint metrics.
