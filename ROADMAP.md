# Roadmap

The application integration is functional; the highest-priority remaining work is **model quality and external validation**, not adding more UI features.

## Completed

- Model audit.
- Face pipeline.
- Local FastAPI inference service.
- Manual extension analysis.
- Visible-page scanning.
- Per-face overlays.
- Prediction cache and deduplication.
- Automated API/extension checks.
- Replaceable model-provider boundary.

## Next — Model Quality

- Build a multi-dataset collection pipeline.
- Add identity-aware or source-aware train/validation/test splits.
- Generate production-style face crops that match browser inputs.
- Train a MobileNetV3Large baseline.
- Run external holdout evaluation.
- Measure robustness under compression, resizing, and screenshots.
- Calibrate scores and uncertainty thresholds.
- Replace the current model through the provider registry.

## Next — Product Validation

- Complete the Phase 5 manual browser test plan.
- Validate the extension on a wider range of websites and image layouts.
- Review error states for missing faces, failed crops, and unavailable model artifacts.

## Later

- Compare Haar with YuNet or MediaPipe.
- Export ONNX or TFLite inference artifacts.
- Add API deployment packaging.
- Prepare Chrome Web Store submission.
- Publish a privacy policy.
- Add production-oriented monitoring and rate limits.

## Research Questions

- Which datasets best represent webpage face crops rather than prepared benchmark faces?
- How much do compression, resizing, screenshots, and social-media processing affect scores?
- Which detector and crop strategy produce the most stable model inputs?
- What threshold policy minimizes harmful false accusations while preserving useful signal?
- Which replacement model gives the best accuracy, latency, size, and calibration tradeoff?
