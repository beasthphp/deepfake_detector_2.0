# Phase 5 Report

## A. Final Architecture

```text
Visible webpage image
        -> Eligibility and deduplication
        -> Bounded request queue
        -> Local API
        -> Cached per-face predictions
        -> Responsive face overlays
```

Manual right-click analysis remains available and uses the same API client/storage path as before.

## B. Permissions

- `contextMenus`: keeps the right-click image analysis command.
- `storage`: stores extension options, manual result history, per-origin scan settings, and prediction cache metadata/results.
- `activeTab`: lets the user-triggered popup action target the current tab.
- `scripting`: injects the content script only when manual analysis notifications or page scanning require it.
- `http://127.0.0.1:8000/*`: allows the default local API.
- `http://*/*` and `https://*/*`: allows fetching user-selected or eligible page images for analysis.

## C. Scanning Controls

Popup controls added:

- `Scan visible images`
- `Stop scanning`
- `Rescan page`
- `Clear page results`
- `Enable scanning on this site`
- `Disable scanning on this site`
- `Clear cache`
- `Emergency stop`
- `Resume scanning`

New sites default to `manual_only`. Site settings are stored per origin as `enabled`, `disabled`, or `manual_only`.

## D. Overlay Geometry

The overlay system maps API bounding boxes from original image pixels onto the displayed image. It accounts for:

- natural image dimensions
- displayed dimensions
- scrolling
- resizing
- browser zoom through CSS pixel geometry
- device pixel ratio independence
- image movement after layout changes
- `object-fit: fill`, `contain`, `cover`, `none`, and `scale-down`
- `object-position` offsets
- clipping when `cover` crops the rendered image content

`ResizeObserver`, throttled scroll/resize handling, and recalculation on dynamic changes keep overlays positioned after layout movement.

## E. Caching And Model Replacement

The API now returns:

- model ID/version
- detector ID/version
- crop-strategy ID/version
- threshold ID/version

Cache keys include:

- image URL
- natural width
- natural height
- model version signature
- detector version signature
- crop-strategy version signature
- threshold version signature
- image-content hash after download where available

Changing any version component invalidates old cached predictions. Raw image bytes and face crops are not stored.

## F. Privacy

The extension:

- sends only user-selected images or eligible images from user-controlled page scans
- sends images only to the configured API origin
- defaults to the local API
- does not store raw image bytes
- does not store face crops
- skips browser-internal pages
- skips password pages for page scanning
- skips `data:` and `blob:` URLs for page scanning
- allows clearing page results and prediction cache

If a remote API is configured, eligible images leave the browser and are transmitted to that remote origin.

## G. Automated Test Results

- Extension tests: `24 passed, 0 failed`
- API tests: `22 passed, 0 failed`
- JavaScript syntax checks: passed
- Manifest parse: passed
- Unsafe rendering scan for `innerHTML`, `eval`, and `document.write`: no matches

See `reports/phase5/AUTOMATED_TEST_RESULTS.md`.

## H. Manual Test Results

Manual browser tests have not been run yet. Results are not fabricated.

See `reports/phase5/MANUAL_TEST_PLAN.md`.

## I. Known Limitations

- Existing model is an experimental placeholder.
- Current real-world predictions are unreliable.
- Haar face detection has known limitations and missed 5% in Phase 2.1 expanded validation.
- Protected images may fail to download.
- Blob, data, CSS background, SVG, canvas, and video content are not scanned by page scanning.
- API cold startup remains noticeable.
- Local API must be running unless mock mode is enabled.
- Scores are uncalibrated model outputs.
- The model has not been externally validated.
- Manual browser testing is still required.

## J. Phase 5 Verdict

Complete extension architecture ready for improved model.

This verdict means the scanner architecture, queue, cache, mock mode, and overlay geometry are implemented and covered by automated tests. It does not mean current predictions are reliable, and it does not replace manual browser verification.

## K. Exact Next Action

Run the Phase 5 manual browser test plan and record actual pass/fail results.
