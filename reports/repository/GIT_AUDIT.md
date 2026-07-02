# Repository Git Audit

Date: 2026-07-02

## Summary

The workspace initially contained a `D:\ddp\.git` directory, but it was not a valid Git repository: `git status`, `git branch --show-current`, `git remote -v`, and `git ls-files` all failed with `fatal: not a git repository`. The `.git` directory had no `HEAD` or refs. Because there was no usable Git history, there were no tracked or committed files to inspect for committed secrets.

After the read-only audit, Git metadata was initialized and the working branch `chore/repository-release-prep` was created. No commit or push has been made.

## Commands Run During Read-Only Audit

- `git status --short --branch`
- `git branch --show-current`
- `git remote -v`
- `git lfs env`
- `git ls-files`
- `git ls-files --others --exclude-standard`
- `git ls-files -oi --exclude-standard`
- `Get-ChildItem -Force`
- `Get-ChildItem -Force -Directory -Recurse -Filter .git`
- `rg --files --hidden -g '!.git/**'`
- Large-file scans with `Get-ChildItem -Force -Recurse -File`
- `.env`, Kaggle, credentials, and secrets filename scans
- Deterministic secret-pattern scans that reported file names only
- Local-path scans for developer-specific absolute paths

## Git State Before Initialization

- Current Git status: unavailable because `.git` was invalid.
- Current branch: unavailable.
- Configured remotes: unavailable.
- Tracked files: unavailable.
- Untracked files: unavailable.
- Ignored files: unavailable.
- Git LFS: installed globally as `git-lfs/3.7.1`, but no repository LFS configuration existed.

## Repository Inventory

- Total files on disk excluding `.git`: 173,718.
- Files under `data/raw`: 140,003.
- Files under `.venv-model-audit`: 33,527.
- Files under `__pycache__`: 9,766.
- Files under `phase2_samples/output`: 23.
- Model artifact extensions found: 7.

## Large Files

### Larger Than 10 MB

- `.venv-model-audit/Lib/site-packages/tensorflow/python/_pywrap_tensorflow_common.dll` - 944.58 MB.
- `models/experiments/retrained_custom_cnn_smoke_best.keras` - 169.87 MB.
- `model/deepfake_detector_93acc.h5` - 169.87 MB.
- `.venv-model-audit/Lib/site-packages/clang/native/libclang.dll` - 80.10 MB.
- `.venv-model-audit/Lib/site-packages/cv2/cv2.pyd` - 71.00 MB.
- `.venv-model-audit/Lib/site-packages/numpy.libs/libopenblas64__*.dll` - 36.40 MB.
- `.venv-model-audit/Lib/site-packages/cv2/opencv_videoio_ffmpeg4100_64.dll` - 25.17 MB.
- `.venv-model-audit/Lib/site-packages/tensorflow/lite/python/pywrap_tflite_common.dll` - 22.67 MB.
- `.venv-model-audit/Lib/site-packages/tensorflow/python/profiler/internal/_pywrap_profiler_plugin.pyd` - 19.50 MB.
- `.venv-model-audit/Lib/site-packages/scipy.libs/libscipy_openblas-*.dll` - 19.32 MB.
- `data/raw/train.csv` - 14.90 MB.
- `.venv-model-audit/Lib/site-packages/tensorflow/compiler/mlir/stablehlo/stablehlo_extension.pyd` - 11.97 MB.
- `.venv-model-audit/Lib/site-packages/tensorflow/compiler/tf2xla/ops/_xla_ops.so` - 10.81 MB.
- `.venv-model-audit/Lib/site-packages/grpc/_cython/cygrpc.cp311-win_amd64.pyd` - 10.39 MB.

### Larger Than 50 MB

- `.venv-model-audit/Lib/site-packages/tensorflow/python/_pywrap_tensorflow_common.dll` - 944.58 MB.
- `models/experiments/retrained_custom_cnn_smoke_best.keras` - 169.87 MB.
- `model/deepfake_detector_93acc.h5` - 169.87 MB.
- `.venv-model-audit/Lib/site-packages/clang/native/libclang.dll` - 80.10 MB.
- `.venv-model-audit/Lib/site-packages/cv2/cv2.pyd` - 71.00 MB.

### Larger Than 100 MB

- `.venv-model-audit/Lib/site-packages/tensorflow/python/_pywrap_tensorflow_common.dll` - 944.58 MB.
- `models/experiments/retrained_custom_cnn_smoke_best.keras` - 169.87 MB.
- `model/deepfake_detector_93acc.h5` - 169.87 MB.

## Possible Sensitive Information

- No `.env`, `.env.*`, `kaggle.json`, `credentials.json`, or `secrets.json` files were found in the workspace scan.
- The deterministic pattern scan outside `.git`, `.venv-model-audit`, `data/raw`, images, and model binaries did not identify candidate secret-bearing files.
- There was no valid Git history, so no secret was found to be already committed.

Secret values were not printed or copied into this report.

## Absolute Local Paths

- `api/tests/test_predict.py` used `D:/ddp/phase2_samples/input` as a fixture path. This is machine-specific and should be changed to a repository-relative path.
- Existing completed audit reports contain historical `D:\ddp` references as evidence from prior runs. Those reports are retained unless a verified correction is required.

## File Classification

### Safe To Commit

- Python source under `api/`, `face_pipeline/`, `training/`, `evaluation/`, and `model/umr.py` where source-only.
- Extension source under `extension/`.
- Test source under `api/tests/`, `face_pipeline/test_pipeline.py`, and `extension/tests/`.
- Requirements files.
- Phase summary documentation under `reports/` except generated prediction CSVs and process logs.
- Sample input images under `phase2_samples/input`.

### Should Be Committed

- `.gitignore`.
- `.env.example`.
- `AGENTS.md`.
- `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `MODEL_CARD.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, and `SECURITY.md`.
- `model_runtime/` provider abstraction.
- `models/README.md`, `models/registry.json`, and `models/artifacts/.gitkeep`.
- `scripts/download_model.py`, `scripts/verify_model.py`, and validation helper scripts.
- `.github/` issue templates, pull request template, and CI workflow.
- `reports/repository/GIT_AUDIT.md`, `GITHUB_ISSUES_PLAN.md`, and `RELEASE_READINESS.md`.

### Should Be Ignored

- `.venv-model-audit/`.
- `.pytest_cache/`.
- `__pycache__/`.
- `data/raw/` and `data/processed/`.
- `phase2_samples/output/` and `phase2_samples/crop_sweep/`.
- `models/experiments/` and `models/artifacts/` model binaries.
- `model/*.h5`.
- Prediction CSV outputs under `reports/**`.
- API process logs and pid/job files under `reports/**`.
- Node coverage and `node_modules/`.
- Local credentials and secret files.

### Should Be Removed From Git Tracking But Retained Locally

No `git rm --cached` commands were needed because the repository had no tracked files at audit time. If this work is later applied to a repository with history, remove these from the index only:

- `.venv-model-audit/`
- `data/raw/`
- `model/deepfake_detector_93acc.h5`
- `models/experiments/*.keras`
- `phase2_samples/output/`
- generated prediction CSVs and server logs
- `__pycache__/` and `.pytest_cache/`

### Requires Git LFS

Only if intentionally versioned as release artifacts:

- `model/deepfake_detector_93acc.h5`
- `models/experiments/retrained_custom_cnn_smoke_best.keras`

The current release-prep approach does not commit model binaries. The model registry points to local or downloadable artifacts instead.

### Possible Sensitive Information

- None detected by filename scan.
- None detected by deterministic pattern scan.
- Manual review should still inspect any future `.env`, Kaggle, credential, or secret files before staging.

### Uncertain And Requiring Manual Review

- `previous_works/` contains legacy source and an old LFS pointer. Keep for provenance unless the project owner chooses to archive it separately.
- Completed report artifacts under `reports/` should be retained, but generated prediction CSVs should remain ignored.
- `model/umr.py` appears to be a notebook-style experimental export and should remain source-only, not evidence of production readiness.

## Exact Git Metadata Commands Used After Audit

- `git init --initial-branch=main`
- `git switch -c chore/repository-release-prep`

No commit, push, merge, tag, destructive cleanup, or history rewrite was performed.
