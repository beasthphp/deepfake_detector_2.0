# Phase 4 Report

## A. Extension Architecture

```text
User-selected webpage image
        ->
Manifest V3 service worker
        ->
Local FastAPI
        ->
Per-face predictions
        ->
Popup and result page
```

The extension is under `extension/`. It creates an image-only context menu item, fetches only the selected image, sends it to the configured local API, persists sanitized state in `chrome.storage.local`, and renders results in the popup and detailed result page.

No automatic page scanning, MutationObserver, IntersectionObserver, or permanent face overlays were added.

## B. Permissions

- `contextMenus`: creates `Analyze face for deepfake` for image context menus only.
- `storage`: stores current request state and latest sanitized analysis history.
- `activeTab`: allows user-initiated interaction with the active page after the context-menu click.
- `scripting`: injects `content-script.js` only after the manual analysis starts, to show temporary page-edge feedback.
- `http://127.0.0.1:8000/*`: allows requests to the local FastAPI service.
- `http://*/*` and `https://*/*`: allows the service worker to fetch arbitrary HTTP/HTTPS images explicitly selected by the user. This is not used for automatic scanning and should be reconsidered before publication.

Permissions not requested: browsing history, tabs, cookies, webRequest.

## C. Supported Image Sources

Implemented:

- `http://` image URLs
- `https://` image URLs
- supported `data:` image URLs, with stored URL redaction

Explicitly handled as unsupported or failure cases:

- `blob:` URLs
- CSS-generated images without a usable image URL
- protected/authenticated images that the extension fetch cannot read
- non-image resources
- empty downloads
- images above the 10 MB API limit

The extension does not fall back to screenshotting the page.

## D. API Integration

Request flow:

1. `GET /health`
2. Fetch selected image bytes.
3. Create `FormData`.
4. Append the image as field `file`.
5. `POST /predict`.
6. Validate the response schema before rendering.

The extension does not manually set multipart `Content-Type`; the browser supplies the boundary.

Handled API outcomes:

- `completed`
- `no_face_detected`
- 400 invalid image
- 413 image too large
- 415 unsupported format
- 422 processing failure
- 429 concurrency rejection
- 500 server failure
- 503 not ready
- 504 timeout
- malformed JSON
- completed response with empty face array

## E. Privacy Behavior

Transmitted:

- only the user-selected image bytes
- only to the configured API base URL

Retained:

- current request status
- selected image URL, with `data:` and `blob:` forms redacted
- page URL
- request ID
- API response
- timestamp
- sanitized error information
- latest history records, default limit 10

Not retained:

- raw image bytes
- base64 image data
- face crops
- screenshots
- browsing history outside manually requested analyses

## F. Test Status

Automated tests completed:

- API URL validation
- selected image URL validation
- data URL redaction
- score formatting
- prediction response schema validation
- no-face rendering summary
- multiple-face summary without averaging
- HTTP error mapping
- safe display text truncation
- storage history limit
- timeout handling
- API health readiness handling
- multipart prediction request construction

Automated result:

```text
13 passed, 0 failed
```

Static checks completed:

- All extension JavaScript files pass `node --check`.
- No `innerHTML`, `eval`, `MutationObserver`, or `IntersectionObserver` usage found.
- Headless Chrome `--load-extension=D:\ddp\extension` smoke check exited with code 0.

Manual browser tests completed:

- None in this Codex session.

Tests not yet completed:

- Loading the extension as unpacked in Chrome/Chromium.
- Confirming service-worker requests against a live API from the real extension origin.
- Right-click analysis of real, fake, multiple-face, no-face, large, protected, data URL, and corrupted-image cases.
- Page refresh and service-worker restart behavior in the browser.

## G. Known Limitations

- The local API must already be running.
- Cold API startup takes about 10.69 seconds.
- Haar face detection has frontal-face limitations.
- Phase 2.1 observed approximately 5% detection/crop failure.
- Image URL fetching may fail for protected, blob, CSS-generated, or unusual image sources.
- Model scores are uncalibrated.
- No automatic scanning is implemented.
- No webpage face overlays are implemented.
- No external real-world validation has been run.
- Manual Chrome/Chromium validation is still required.

## H. Phase 4 Verdict

Controlled extension works at the implementation and automated-test level but needs manual browser validation before it should be treated as ready.

Selected verdict category: `controlled extension works but needs fixes`.

The remaining required fix is validation in a real unpacked Chrome/Chromium session against the local API. No automatic scanning prototype should start until that manual pass is complete.

## I. Exact Next Action

Load `D:\ddp\extension` as an unpacked Chrome extension and run the manual test plan against the local API.
