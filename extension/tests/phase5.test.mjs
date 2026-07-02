import assert from "node:assert/strict";
import test from "node:test";

import { checkHealth, ExtensionRequestError } from "../shared/api-client.js";
import { createCacheEntry, isCacheEntryFresh } from "../shared/cache.js";
import { MOCK_SCENARIOS, SITE_MODE } from "../shared/constants.js";
import { predictMockImage } from "../shared/mock-api.js";
import { computeRenderedImageGeometry, mapFaceBoxToDisplayedImage } from "../shared/overlay-geometry.js";
import { BoundedRequestQueue } from "../shared/request-queue.js";
import {
  backendMetadataFromHealth,
  buildImageContentKey,
  buildImageSourceKey,
  evaluateImageEligibility,
  getSiteMode,
  isStaleScanResult,
  setSiteMode
} from "../shared/scanner.js";
import { buildResultSummary, normalizeOptions, safeDisplayText, validatePredictionResponse } from "../shared/validation.js";

const metadataV1 = backendMetadataFromHealth({
  model_id: "existing-cnn-93acc",
  model_version: "1",
  detector_id: "opencv_haar_frontalface_default",
  detector_version: "1",
  crop_strategy_id: "preserve_or_context_full_head_l40_r40_t70_b35_square",
  crop_strategy_version: "phase2.1",
  threshold_id: "score-thresholds",
  threshold_version: "1"
});

const eligibleDescriptor = {
  id: "img-1",
  currentSrc: "https://example.com/photo.jpg",
  naturalWidth: 800,
  naturalHeight: 600,
  displayedWidth: 400,
  displayedHeight: 300,
  isLoaded: true,
  isVisible: true,
  visibilityState: "visible"
};

test("image eligibility filters loaded visible images and skips small/icon inputs", () => {
  assert.equal(evaluateImageEligibility(eligibleDescriptor, {}, new Set(), metadataV1).eligible, true);
  assert.equal(
    evaluateImageEligibility({ ...eligibleDescriptor, displayedWidth: 64 }, {}, new Set(), metadataV1).reason,
    "display_too_small"
  );
  assert.equal(
    evaluateImageEligibility(
      { ...eligibleDescriptor, currentSrc: "https://example.com/assets/logo.png", naturalWidth: 256, naturalHeight: 256 },
      { minDisplayedWidth: 128, minDisplayedHeight: 128, minNaturalWidth: 128, minNaturalHeight: 128 },
      new Set(),
      metadataV1
    ).reason,
    "likely_icon_or_logo"
  );
});

test("source and content keys include source dimensions and backend versions", () => {
  const first = buildImageSourceKey(eligibleDescriptor, metadataV1);
  const same = buildImageSourceKey({ ...eligibleDescriptor, id: "img-2" }, metadataV1);
  const changedSource = buildImageSourceKey({ ...eligibleDescriptor, currentSrc: "https://example.com/other.jpg" }, metadataV1);
  const content = buildImageContentKey(eligibleDescriptor, metadataV1, "abc123");
  assert.equal(first, same);
  assert.notEqual(first, changedSource);
  assert.match(content, /abc123/);
});

test("cache entries are invalidated by backend version changes", () => {
  const entry = createCacheEntry({
    sourceKey: buildImageSourceKey(eligibleDescriptor, metadataV1),
    status: "completed",
    response: { status: "completed" },
    metadata: metadataV1,
    now: 1000
  });
  const metadataV2 = backendMetadataFromHealth({ model_id: "existing-cnn-93acc", model_version: "2" });
  assert.equal(isCacheEntryFresh(entry, metadataV1, 2000), true);
  assert.equal(isCacheEntryFresh(entry, metadataV2, 2000), false);
});

test("bounded request queue respects concurrency", async () => {
  let active = 0;
  let maxActive = 0;
  let completed = 0;
  const queue = new BoundedRequestQueue({
    concurrency: 2,
    maxQueued: 10,
    worker: async () => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await delay(20);
      active -= 1;
      completed += 1;
    }
  });
  for (let index = 0; index < 5; index += 1) {
    assert.equal(queue.enqueue({ id: `item-${index}` }).accepted, true);
  }
  await waitUntil(() => completed === 5);
  assert.equal(maxActive, 2);
});

test("bounded request queue cancels queued work", async () => {
  const started = [];
  const queue = new BoundedRequestQueue({
    concurrency: 1,
    maxQueued: 5,
    worker: async (item) => {
      started.push(item.id);
      await delay(30);
    }
  });
  queue.enqueue({ id: "active" });
  queue.enqueue({ id: "queued-1" });
  queue.enqueue({ id: "queued-2" });
  queue.cancelWhere((item) => item.id.startsWith("queued"));
  await delay(80);
  assert.deepEqual(started, ["active"]);
});

test("object-fit contain geometry preserves letterbox offsets", () => {
  const geometry = computeRenderedImageGeometry({
    naturalWidth: 400,
    naturalHeight: 200,
    displayedWidth: 200,
    displayedHeight: 200,
    objectFit: "contain",
    objectPosition: "50% 50%"
  });
  assert.equal(geometry.scaleX, 0.5);
  assert.equal(geometry.offsetY, 50);
  const mapped = mapFaceBoxToDisplayedImage({ x1: 0, y1: 0, x2: 100, y2: 100 }, geometry);
  assert.deepEqual({ x: mapped.x, y: mapped.y, width: mapped.width, height: mapped.height }, { x: 0, y: 50, width: 50, height: 50 });
});

test("object-fit cover geometry clips cropped regions", () => {
  const geometry = computeRenderedImageGeometry({
    naturalWidth: 400,
    naturalHeight: 200,
    displayedWidth: 200,
    displayedHeight: 200,
    objectFit: "cover",
    objectPosition: "50% 50%"
  });
  assert.equal(geometry.offsetX, -100);
  const mapped = mapFaceBoxToDisplayedImage({ x1: 0, y1: 0, x2: 160, y2: 200 }, geometry);
  assert.equal(mapped.clipped, true);
  assert.deepEqual({ x: mapped.x, y: mapped.y, width: mapped.width, height: mapped.height }, { x: 0, y: 0, width: 60, height: 200 });
});

test("object-position offsets are applied", () => {
  const bottom = computeRenderedImageGeometry({
    naturalWidth: 100,
    naturalHeight: 50,
    displayedWidth: 200,
    displayedHeight: 200,
    objectFit: "contain",
    objectPosition: "right bottom"
  });
  assert.equal(bottom.offsetX, 0);
  assert.equal(bottom.offsetY, 100);
});

test("no-face and multiple-face responses keep image-level and per-face states separate", async () => {
  const noFace = await predictMockImage({ scenario: MOCK_SCENARIOS.noFace, imageBlob: new Blob(["x"], { type: "image/jpeg" }) });
  const multiple = await predictMockImage({ scenario: MOCK_SCENARIOS.multipleFaces, imageBlob: new Blob(["x"], { type: "image/jpeg" }) });
  assert.equal(validatePredictionResponse(noFace).ok, true);
  assert.equal(noFace.status, "no_face_detected");
  assert.equal(validatePredictionResponse(multiple).ok, true);
  assert.match(buildResultSummary(multiple), /1 Likely Fake/);
  assert.match(buildResultSummary(multiple), /1 Uncertain/);
  assert.match(buildResultSummary(multiple), /1 Likely Real/);
});

test("site modes, global stop normalization, stale results, API offline, and safe text are handled", async () => {
  const settings = setSiteMode({}, "https://example.com/page", SITE_MODE.enabled);
  assert.equal(getSiteMode(settings, "https://example.com/other"), SITE_MODE.enabled);
  assert.equal(normalizeOptions({ globalScanningStopped: true }).globalScanningStopped, true);
  assert.equal(
    isStaleScanResult(
      { sessionId: "old", imageId: "img", sourceKey: "source" },
      { sessionId: "new", imageId: "img", sourceKey: "source" }
    ),
    true
  );
  await assert.rejects(
    () =>
      checkHealth(
        "http://127.0.0.1:8000",
        100,
        async () => {
          throw new TypeError("offline");
        }
      ),
    ExtensionRequestError
  );
  assert.equal(safeDisplayText("hello\u0000\nworld"), "hello world");
});

test("mock malformed and API-error scenarios are deterministic", async () => {
  const malformed = await predictMockImage({ scenario: MOCK_SCENARIOS.malformed, imageBlob: new Blob(["x"], { type: "image/jpeg" }) });
  assert.equal(validatePredictionResponse(malformed).ok, false);
  await assert.rejects(
    () => predictMockImage({ scenario: MOCK_SCENARIOS.apiError, imageBlob: new Blob(["x"], { type: "image/jpeg" }) }),
    { code: "mock_api_error" }
  );
});

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitUntil(predicate, timeoutMs = 1000) {
  const started = Date.now();
  while (!predicate()) {
    if (Date.now() - started > timeoutMs) {
      throw new Error("Timed out waiting for condition.");
    }
    await delay(5);
  }
}
