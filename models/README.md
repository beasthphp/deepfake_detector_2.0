# Model Artifacts

Actual model binaries are not committed to this repository.

The active model is selected by `ACTIVE_MODEL_ID` and described in `models/registry.json`. By default, the legacy MVP classifier expects:

```text
models/artifacts/deepfake_detector_93acc.h5
```

If no download URL is configured, place the model manually in `models/artifacts/` or set `DEEPFAKE_MODEL_PATH` to an absolute or repository-relative path:

```powershell
$env:DEEPFAKE_MODEL_PATH="model/deepfake_detector_93acc.h5"
python scripts/verify_model.py --model-id legacy-cnn-v1
```

The provider rejects missing files, Git LFS pointer files, unexpected file types, hash mismatches when a hash is configured, and input/output shape mismatches.
