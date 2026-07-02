# Extension Setup

## Start The Local API

From the project root:

```powershell
& 'D:\ddp\.venv-model-audit\Scripts\python.exe' -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Keep the API local. Do not expose the development server publicly.

Verify health:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -Method Get
```

Expected response includes:

```json
{
  "status": "ok",
  "model_loaded": true,
  "detector_loaded": true
}
```

## Load The Extension

1. Open Chrome or Chromium.
2. Go to `chrome://extensions`.
3. Enable `Developer mode`.
4. Click `Load unpacked`.
5. Select `D:\ddp\extension`.
6. Confirm the extension appears as `Local Deepfake Face Analyzer`.

## Find The Extension ID

On `chrome://extensions`, the extension card shows an `ID` value. Copy it if the API needs an explicit CORS origin.

If CORS blocks requests, start the API with the extension origin included:

```powershell
$env:ALLOWED_ORIGINS='http://localhost,http://127.0.0.1,chrome-extension://<extension-id>'
& 'D:\ddp\.venv-model-audit\Scripts\python.exe' -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Do not disable CORS globally and do not use unrestricted credentialed CORS.

## Analyze A Webpage Image

1. Open a webpage with a visible image.
2. Right-click the image.
3. Choose `Analyze face for deepfake`.
4. The extension checks the local API, downloads only the selected image, sends it to `/predict`, and stores the response.
5. Open the toolbar popup or result page to view per-face results.

The extension does not scan the page automatically and does not analyze unrelated images.

## Inspect Extension Errors

1. Open `chrome://extensions`.
2. Find `Local Deepfake Face Analyzer`.
3. Click `service worker` or `Inspect views` if shown.
4. Check console errors.
5. Check the popup and results pages separately by right-clicking them and choosing `Inspect`.

## Options

The options page allows:

- API base URL
- Request timeout
- Automatic result-page opening
- Score visibility
- Recent-history limit

Default API base URL is `http://127.0.0.1:8000`.
