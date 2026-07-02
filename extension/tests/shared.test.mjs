import assert from "node:assert/strict";
import test from "node:test";

import { fetchWithTimeout, predictImage, checkHealth, ExtensionRequestError } from "../shared/api-client.js";
import { trimHistory } from "../shared/storage.js";
import {
  buildResultSummary,
  formatPercent,
  mapApiError,
  redactImageUrlForStorage,
  safeDisplayText,
  validateApiBaseUrl,
  validatePredictionResponse,
  validateSelectedImageUrl
} from "../shared/validation.js";

const completedResponse = {
  request_id: "req-1",
  status: "completed",
  image: { width: 256, height: 256, format: "JPEG" },
  faces_detected: 2,
  faces: [
    {
      face_index: 0,
      bounding_box: { x1: 1, y1: 2, x2: 3, y2: 4 },
      crop_box: { x1: 0, y1: 0, x2: 10, y2: 10 },
      face_detection_score: 0.91,
      crop_strategy: "preserve_or_context_full_head_l40_r40_t70_b35_square",
      preserved_original: true,
      real_score: 0.12,
      fake_score: 0.88,
      label: "Likely Fake"
    },
    {
      face_index: 1,
      bounding_box: { x1: 11, y1: 12, x2: 13, y2: 14 },
      crop_box: { x1: 10, y1: 10, x2: 20, y2: 20 },
      face_detection_score: 0.84,
      crop_strategy: "context_full_head_l40_r40_t70_b35_square",
      preserved_original: false,
      real_score: 0.52,
      fake_score: 0.48,
      label: "Uncertain"
    }
  ],
  timing_ms: { decode: 1, face_detection: 2, crop_preprocessing: 3, classification: 4, serialization: 1, total: 11 },
  warnings: []
};

test("API URL validation accepts local default and warns on public HTTP", () => {
  assert.equal(validateApiBaseUrl("http://127.0.0.1:8000").ok, true);
  assert.equal(validateApiBaseUrl("ftp://127.0.0.1:8000").ok, false);
  const publicHttp = validateApiBaseUrl("http://example.com:8000");
  assert.equal(publicHttp.ok, true);
  assert.match(publicHttp.warnings[0], /Public HTTP/);
});

test("selected image URL validation supports http, https, and image data URLs only", () => {
  assert.equal(validateSelectedImageUrl("https://example.com/image.jpg").ok, true);
  assert.equal(validateSelectedImageUrl("http://example.com/image.png").ok, true);
  assert.equal(validateSelectedImageUrl("data:image/png;base64,AAAA").ok, true);
  assert.equal(validateSelectedImageUrl("blob:https://example.com/abc").code, "unsupported_blob_url");
  assert.equal(validateSelectedImageUrl("file:///tmp/image.jpg").ok, false);
});

test("data URLs are redacted before storage", () => {
  assert.equal(redactImageUrlForStorage("data:image/png;base64,SECRET"), "data:image/png;<redacted>");
});

test("score formatting rounds to percentages", () => {
  assert.equal(formatPercent(0.876), "88%");
  assert.equal(formatPercent(0.001), "0%");
  assert.equal(formatPercent(Number.NaN), "n/a");
});

test("prediction schema validation accepts completed and no-face responses", () => {
  assert.equal(validatePredictionResponse(completedResponse).ok, true);
  assert.equal(
    validatePredictionResponse({
      request_id: "req-2",
      status: "no_face_detected",
      image: { width: 320, height: 240, format: "JPEG" },
      faces_detected: 0,
      faces: [],
      timing_ms: {},
      warnings: []
    }).ok,
    true
  );
});

test("prediction schema validation rejects completed responses without faces", () => {
  const invalid = { ...completedResponse, faces_detected: 0, faces: [] };
  assert.equal(validatePredictionResponse(invalid).code, "completed_without_faces");
});

test("summary logic preserves per-face categories without averaging", () => {
  assert.match(buildResultSummary(completedResponse), /1 Likely Fake/);
  assert.match(buildResultSummary(completedResponse), /1 Uncertain/);
  assert.match(buildResultSummary(completedResponse), /0 Likely Real/);
  assert.equal(
    buildResultSummary({ status: "no_face_detected", faces_detected: 0, faces: [] }),
    "No supported human face was detected."
  );
});

test("HTTP error mapping uses structured server errors safely", () => {
  const mapped = mapApiError(413, { error: { code: "upload_too_large", message: "Too large" } });
  assert.equal(mapped.code, "upload_too_large");
  assert.equal(mapped.message, "Too large");
  assert.equal(mapApiError(429, {}).message, "The local detector is busy. Try again in a moment.");
});

test("safe display text removes control characters and truncates", () => {
  const value = safeDisplayText("hello\u0000\nworld".repeat(20), 24);
  assert.equal(value.includes("\u0000"), false);
  assert.ok(value.length <= 24);
});

test("storage history limit keeps only the latest records", () => {
  const history = Array.from({ length: 12 }, (_, index) => ({ analysisId: String(index) }));
  assert.deepEqual(trimHistory(history, 3).map((item) => item.analysisId), ["0", "1", "2"]);
});

test("fetch timeout rejects with AbortError", async () => {
  const slowFetch = (_url, options) =>
    new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    });
  await assert.rejects(() => fetchWithTimeout("http://example.com", {}, 1, slowFetch), { name: "AbortError" });
});

test("health check rejects API that is not fully loaded", async () => {
  const fakeFetch = async () =>
    new Response(JSON.stringify({ status: "starting", model_loaded: false, detector_loaded: true }), {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  await assert.rejects(() => checkHealth("http://127.0.0.1:8000", 100, fakeFetch), ExtensionRequestError);
});

test("predictImage posts multipart form without manually setting Content-Type", async () => {
  let observedOptions;
  const fakeFetch = async (_url, options) => {
    observedOptions = options;
    return new Response(JSON.stringify(completedResponse), {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  };

  const blob = new Blob(["abc"], { type: "image/jpeg" });
  const result = await predictImage("http://127.0.0.1:8000", blob, "image.jpg", 1000, fakeFetch);
  assert.equal(result.status, "completed");
  assert.equal(observedOptions.method, "POST");
  assert.equal(observedOptions.headers, undefined);
  assert.ok(observedOptions.body instanceof FormData);
});
