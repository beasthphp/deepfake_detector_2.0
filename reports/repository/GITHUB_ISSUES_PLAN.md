# GitHub Issues Plan

Suggested milestone for these issues: `0.2.0 model replacement`, unless otherwise noted.

## 1. Complete Phase 5 manual browser test plan

Description: Run the documented manual browser tests for manual image analysis, visible-page scanning, overlays, cache behavior, site controls, and emergency stop/resume.

Acceptance criteria: Manual test results are recorded with pass/fail status, browser version, API mode, screenshots where useful, and any defects opened separately.

Labels: `extension`, `testing`, `manual`

Dependencies: Current local API and extension build.

Suggested milestone: `0.1.0 release readiness`

## 2. Build multi-dataset manifest pipeline

Description: Define a manifest format that can describe image path, label, source dataset, split, identity or source group when available, license notes, and preprocessing provenance.

Acceptance criteria: Manifest builder can ingest at least two dataset sources and produce deterministic train/valid/test manifests without copying raw files.

Labels: `data`, `model`, `research`

Dependencies: Dataset access decisions.

Suggested milestone: `0.2.0 model replacement`

## 3. Add identity-aware or source-aware splits

Description: Prevent leakage by grouping examples by identity, source video, generator, or dataset source when labels are available.

Acceptance criteria: Split code enforces group separation, reports group counts, and fails when required group metadata is missing.

Labels: `data`, `evaluation`, `model`

Dependencies: Multi-dataset manifest pipeline.

Suggested milestone: `0.2.0 model replacement`

## 4. Train MobileNetV3Large baseline

Description: Train a transfer-learning baseline using clean manifests and shared preprocessing.

Acceptance criteria: Training script records config, seed, model version, metrics, confusion matrix, ROC-AUC, false-positive rate, false-negative rate, and saved artifact metadata.

Labels: `model`, `training`

Dependencies: Manifest pipeline and leak-aware splits.

Suggested milestone: `0.2.0 model replacement`

## 5. Add webpage degradation augmentations

Description: Add augmentations for resize, crop, JPEG recompression, screenshots, blur, sharpening, and mild occlusion that mimic web images.

Acceptance criteria: Augmentation policy is configurable, deterministic under seed, documented, and covered by visual spot checks.

Labels: `model`, `data`, `augmentation`

Dependencies: Baseline training pipeline.

Suggested milestone: `0.2.0 model replacement`

## 6. Evaluate on external datasets

Description: Evaluate candidate models against holdout datasets not used for training or threshold selection.

Acceptance criteria: Report includes dataset descriptions, metrics, score distributions, calibration plots, and failure examples.

Labels: `evaluation`, `model`, `research`

Dependencies: Candidate model and external manifests.

Suggested milestone: `0.2.0 model replacement`

## 7. Calibrate uncertainty thresholds

Description: Develop a threshold profile from validation and external holdout evidence.

Acceptance criteria: Threshold profile includes likely fake, uncertain, likely real ranges, calibration evidence, and documented risks.

Labels: `calibration`, `model`, `evaluation`

Dependencies: External evaluation.

Suggested milestone: `0.2.0 model replacement`

## 8. Compare Haar with YuNet or MediaPipe

Description: Compare face-detection recall, crop stability, speed, and integration complexity.

Acceptance criteria: Report covers detector metrics, missed-face examples, crop differences, and recommendation.

Labels: `face-detection`, `research`, `pipeline`

Dependencies: Representative webpage and dataset samples.

Suggested milestone: `0.3.0 deployment hardening`

## 9. Export optimized inference model

Description: Export the selected replacement model to a smaller inference format.

Acceptance criteria: Exported artifact preserves expected scores within tolerance, passes provider verification, and has documented size and latency.

Labels: `model`, `optimization`, `deployment`

Dependencies: Replacement model selection.

Suggested milestone: `0.3.0 deployment hardening`

## 10. Replace legacy model using provider registry

Description: Add a new registry entry and provider settings for the selected replacement model.

Acceptance criteria: API starts with the new active model, `/predict` contract remains compatible, extension cache invalidates by model version, and legacy model remains available as disabled or explicitly selectable if retained.

Labels: `model`, `api`, `release`

Dependencies: External evaluation and calibration.

Suggested milestone: `0.2.0 model replacement`

## 11. Add Docker deployment

Description: Package the local API for reproducible containerized execution.

Acceptance criteria: Docker build excludes datasets and model binaries, accepts mounted model artifacts, exposes documented environment variables, and passes mocked tests.

Labels: `deployment`, `api`, `devops`

Dependencies: Stable model artifact handling.

Suggested milestone: `0.3.0 deployment hardening`

## 12. Prepare Chrome Web Store privacy documentation

Description: Draft the extension privacy disclosures and data-use documentation.

Acceptance criteria: Documentation describes local API defaults, remote API implications, stored cache fields, raw image handling, permissions, and user controls.

Labels: `extension`, `privacy`, `release`

Dependencies: Manual browser test completion and final extension permissions review.

Suggested milestone: `0.3.0 deployment hardening`
