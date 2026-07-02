# Manual Test Plan

Manual browser testing is required before using this as a real workflow. Do not mark a case complete unless it has been run in Chrome or Chromium with the unpacked extension loaded.

| ID | Case | Steps | Expected Result | Status |
| --- | --- | --- | --- | --- |
| 1 | API offline | Stop the API, right-click an image, choose analysis. | Popup/results show `Local detector is not running` and startup command without developer-specific paths. | Not run |
| 2 | API online | Start API, right-click a normal image. | Health check passes, request progresses beyond API check. | Not run |
| 3 | Clear real face | Right-click a prepared clear real-face sample hosted on a page. | `/predict` completes and one face result renders without treating scores as proof. | Not run |
| 4 | Clear fake face | Right-click a prepared clear fake-face sample hosted on a page. | `/predict` completes and one face result renders with `Likely Fake`, `Uncertain`, or `Likely Real` exactly as API returns. | Not run |
| 5 | Multiple faces | Right-click a multi-face image. | Every returned face renders separately; no average image verdict is shown. | Not run |
| 6 | No face | Right-click a non-face image. | Shows `No supported human face was detected. The deepfake classifier was not run. This does not mean the image is real.` | Not run |
| 7 | Large image | Right-click an image larger than the API size limit. | Extension or API reports image too large; no misleading result label. | Not run |
| 8 | Protected image | Right-click an authenticated or protected image. | Download failure is reported safely. | Not run |
| 9 | Data URL | Right-click an image with a supported `data:` URL. | Data image is decoded if Chrome supplies it as `srcUrl`; stored URL is redacted. | Not run |
| 10 | Corrupted or non-image resource | Right-click or simulate an image URL that returns non-image bytes. | Extension/API reports unsupported or invalid image. | Not run |
| 11 | Page refresh during analysis | Start analysis and refresh the page. | Popup/results recover from persisted storage state; page toast may disappear. | Not run |
| 12 | Service-worker restart | Start analysis, wait, inspect service worker lifecycle. | Stored current request and history remain available. | Not run |
| 13 | Clear history | Click clear in popup or results page. | Current request and recent history are removed. | Not run |
| 14 | Change API URL | Open options, change API URL, save. | URL validates; public HTTP endpoint warning appears when applicable. | Not run |

## Automated Checks Already Run

```powershell
node --test extension\tests\*.test.mjs
```

Result: 13 passed, 0 failed.

Static checks:

- All extension JavaScript files pass `node --check`.
- No `innerHTML`, `eval`, `MutationObserver`, or `IntersectionObserver` usage was found.
- A headless Chrome `--load-extension=D:\ddp\extension` smoke check exited with code 0.

This headless check does not replace loading the extension through `chrome://extensions` and using the right-click flow manually.
