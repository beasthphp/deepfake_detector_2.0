# Local Deepfake Face Analyzer Extension

This is the Phase 5 Chrome/Chromium extension prototype.

It supports two user-controlled flows:

```text
Right-click webpage image
    -> Analyze face for deepfake
    -> Local FastAPI /predict
    -> Popup/results page
```

```text
Open popup
    -> Scan visible images
    -> Eligible visible/near-viewport images are queued
    -> Local FastAPI /predict
    -> Per-face overlays on detected faces
```

The current classifier is an experimental placeholder. Real-world predictions are not reliable or externally validated.

## Defaults

Default API endpoint:

```text
http://127.0.0.1:8000
```

New sites default to `manual_only`. Page scanning starts only after a user action or after the user enables the current site. Manual right-click analysis remains available.

## Controls

The popup includes:

- scan visible images
- stop scanning
- rescan page
- clear page results
- enable scanning on this site
- disable scanning on this site
- clear prediction cache
- emergency stop
- resume scanning

The options page includes scan thresholds, queue/concurrency limits, cache size, overlays, diagnostics, mock mode, and API settings.

## Privacy

The extension does not store raw image bytes or face crops. With the default local API, eligible images are sent only to `http://127.0.0.1:8000`. If a remote API is configured, eligible images are sent to that configured remote origin.

Page scanning skips browser-internal pages, password pages, `data:` URLs, `blob:` URLs, CSS background images, SVG elements, canvas content, and videos.

## Development Checks

Latest automated results:

```text
Extension tests: 24 passed, 0 failed
API tests: 22 passed, 0 failed
JavaScript syntax checks: passed
Manifest parse: passed
Unsafe rendering scan: no innerHTML/eval/document.write matches
```

See `reports/phase5` for setup, automated results, manual test plan, and the Phase 5 report.
