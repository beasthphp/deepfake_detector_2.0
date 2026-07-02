# Phase 5 Setup

## API

Start the local FastAPI service from the repository root:

```powershell
.\.venv-model-audit\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Default extension API origin:

```text
http://127.0.0.1:8000
```

The `/predict` response now includes model, detector, crop-strategy, and threshold metadata so extension cache entries can be invalidated when the backend contract changes.

## Extension

Load `D:\ddp\extension` as an unpacked Chrome/Chromium extension from `chrome://extensions`.

Preserved manual flow:

```text
Right-click webpage image
    -> Analyze face for deepfake
    -> Local API /predict
    -> Popup/results page
```

New page-scanning flow:

```text
Open popup
    -> Scan visible images
    -> Eligible visible/near-viewport images are queued
    -> Local API /predict
    -> Per-face overlays are drawn over detected faces
```

Scanning defaults:

- New sites are `manual_only`.
- No global automatic scanning is enabled.
- `Enable scanning on this site` stores the current origin as enabled.
- `Disable scanning on this site` stores the current origin as disabled and stops page scanning there.
- `Emergency stop` disables scanning globally until resumed.

## Options

Configurable settings include:

- API base URL and timeout
- minimum displayed and natural image dimensions
- near-viewport scan margin
- scan request concurrency
- scan queue limit
- prediction cache limit
- page overlays
- diagnostics panel
- development mock API mode and scenario
- global emergency stop

Mock mode is visibly marked and is intended only for deterministic extension UI/overlay testing.

## Privacy Note

With the default local API, eligible selected/scanned image bytes are sent only to `http://127.0.0.1:8000`. If a remote API is configured, selected eligible webpage images leave the browser and are sent to that configured origin. The extension does not store raw image bytes or face crops.
