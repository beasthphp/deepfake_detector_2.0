# Model artifacts and registry

Model binaries are intentionally not committed to this repository.

The active detector is selected through `ACTIVE_MODEL_ID` and described in `models/registry.json`. The registry keeps model identity, provider configuration, and artifact expectations separate from the API and browser extension.

By default, the legacy MVP classifier expects an artifact at:

```text
models/artifacts/deepfake_detector_93acc.h5
```

If no download URL is configured, place the model manually in `models/artifacts/` or set `DEEPFAKE_MODEL_PATH` to an absolute or repository-relative path:

```powershell
$env:DEEPFAKE_MODEL_PATH="model/deepfake_detector_93acc.h5"
python scripts/verify_model.py --model-id legacy-cnn-v1
```

The provider validates missing files, Git LFS pointer files, unexpected file types, configured hash mismatches, and incompatible model input/output shapes before inference begins.

See `MODEL_CARD.md` for the verified architecture, reproduced metrics, known validation limitations, and intended-use boundaries of the current model.
