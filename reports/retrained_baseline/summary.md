# Retrained Baseline Stop Summary

Date: 2026-07-02

## Status

- Clean training pipeline created: yes
- Original files preserved: yes
- Smoke test passed: yes
- Full training completed: no
- Reason full training was not launched: TensorFlow detected CPU only. The smoke fit took 32.43 seconds for 1,000 training images plus 200 validation images. Extrapolating from the observed per-step timing, full training on 100,000 training images plus 20,000 validation images would likely take roughly 45 to 60 minutes per epoch on this CPU path, before early stopping.

## Verified Split Counts

| Split | Fake | Real |
| --- | ---: | ---: |
| train.csv | 50,000 | 50,000 |
| valid.csv | 10,000 | 10,000 |

`validation_split` is not used by the clean pipeline.

## Smoke Test

| Item | Value |
| --- | --- |
| Train subset | 500 fake, 500 real |
| Validation subset | 100 fake, 100 real |
| Batch size | 32 |
| Epochs | 1 |
| Augmentation | enabled, training only |
| Train image validation | 1,000 checked, 0 failed |
| Validation image validation | 200 checked, 0 failed |
| Checkpoint | `D:\ddp\models\experiments\retrained_custom_cnn_smoke_best.keras` |
| Best validation epoch | 1 |
| Best validation ROC-AUC | 0.685700 |

Batch verification:

- Train batch shape: `[32, 256, 256, 3]`
- Validation batch shape: `[32, 256, 256, 3]`
- Image dtype: `float32`
- Pixel range: `0.0` to `1.0`
- Train batch labels: `{"0": 12, "1": 20}`
- Validation batch labels: `{"0": 18, "1": 14}`

Smoke metrics:

| Metric | Value |
| --- | ---: |
| train accuracy | 0.484000 |
| train ROC-AUC | 0.480808 |
| train loss | 0.935248 |
| validation accuracy | 0.570000 |
| validation ROC-AUC | 0.685700 |
| validation loss | 0.690659 |
| validation precision | 0.750000 |
| validation recall | 0.210000 |

These smoke metrics only verify the pipeline; they are not a useful model-quality estimate.

## Full Training Command

When a GPU-capable TensorFlow environment is available, run:

```powershell
& 'D:\ddp\.venv-model-audit\Scripts\python.exe' -m training.train_baseline --full --batch-size 32
```

If memory is insufficient, use:

```powershell
& 'D:\ddp\.venv-model-audit\Scripts\python.exe' -m training.train_baseline --full --batch-size 16
```

Then evaluate the best checkpoint:

```powershell
& 'D:\ddp\.venv-model-audit\Scripts\python.exe' -m training.evaluate --batch-size 32
```

## Recommendation

Do not proceed to MobileNetV3Large yet. First run the full corrected custom-CNN baseline in a GPU-capable environment and evaluate `models/experiments/retrained_custom_cnn_best.keras` on `test.csv`.
