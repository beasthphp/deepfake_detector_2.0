# Crop Consistency Report

## Method

- Dataset: `data/raw/test.csv` with images under `D:\ddp\data\raw\real_vs_fake\real-vs-fake`
- Deterministic seed: `69`
- Accepted images: 50 total (25 real, 25 fake)
- Detector: `opencv_haar_frontalface_default`
- Comparison: original prepared 256x256 image prediction vs. detector crop prediction.
- Class-change threshold: `real_score >= 0.5` means real, otherwise fake.

## Summary

- Average absolute score difference: 0.328043
- Median absolute score difference: 0.057793
- Max absolute score difference: 0.999936
- Predicted class changes: 16 / 50
- Uncertainty increased: 27 / 50
- Skipped images before reaching target: 2

## By Class

| True class | Count | Avg abs diff | Median abs diff | Class changes | Uncertainty increased |
| --- | ---: | ---: | ---: | ---: | ---: |
| real | 25 | 0.090990 | 0.014527 | 2 | 6 |
| fake | 25 | 0.565096 | 0.865071 | 14 | 21 |

## Performance

- Average face-detection time: 27.205 ms
- Average recropped classification time per face: 166.703 ms
- Average total processing time per image: 371.268 ms
- Runtime: CPU TensorFlow in the existing model-audit environment.

## Interpretation

Automatic recropping materially changes model behavior. The detector and crop settings are usable for plumbing tests, but the crop pipeline should be adjusted before API integration.

Full row-level data is saved in `reports/phase2/crop_consistency.json`.
