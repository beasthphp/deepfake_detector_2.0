# Roadmap

## Completed

- Model audit.
- Face pipeline.
- API.
- Manual extension analysis.
- Page scanning.
- Overlays.
- Cache.
- Tests.

## Next

- Complete Phase 5 manual browser test plan.
- Build multi-dataset collection pipeline.
- Add identity-aware or source-aware splits.
- Generate production-style face crops.
- Train MobileNetV3Large baseline.
- Run external holdout evaluation.
- Calibrate scores and uncertainty thresholds.
- Replace current model through the provider registry.

## Later

- Compare Haar with YuNet or MediaPipe.
- Export ONNX or TFLite inference artifacts.
- Add API deployment packaging.
- Prepare Chrome Web Store submission.
- Publish privacy policy.
- Add monitoring and rate limits.

## Research Questions

- Which datasets best represent webpage face crops rather than prepared benchmark faces?
- How much do compression, resizing, screenshots, and social-media processing affect scores?
- Which detector and crop strategy produce the most stable model inputs?
- What threshold policy minimizes harmful false accusations while preserving useful signal?
- Which replacement model gives the best accuracy, latency, size, and calibration tradeoff?
