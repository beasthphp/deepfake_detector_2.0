import {
  DEFAULT_OPTIONS,
  MAX_UPLOAD_BYTES,
  MOCK_SCENARIOS,
  SUPPORTED_IMAGE_EXTENSIONS,
  SUPPORTED_IMAGE_TYPES
} from "./constants.js";

export function validateApiBaseUrl(value) {
  const raw = String(value || "").trim();
  if (!raw) {
    return { ok: false, code: "api_url_empty", message: "API URL is required." };
  }

  let url;
  try {
    url = new URL(raw);
  } catch {
    return { ok: false, code: "api_url_invalid", message: "API URL is not a valid URL." };
  }

  if (!["http:", "https:"].includes(url.protocol)) {
    return {
      ok: false,
      code: "api_url_unsupported_scheme",
      message: "API URL must use http:// or https://."
    };
  }

  if (url.username || url.password) {
    return {
      ok: false,
      code: "api_url_credentials_not_allowed",
      message: "API URL must not include credentials."
    };
  }

  const normalized = removeTrailingSlash(url.href);
  const warnings = [];
  if (url.protocol === "http:" && !isLocalApiHost(url.hostname)) {
    warnings.push("Public HTTP API endpoints are not recommended for this prototype.");
  }

  return { ok: true, value: normalized, warnings };
}

export function normalizeOptions(input = {}) {
  const apiValidation = validateApiBaseUrl(input.apiBaseUrl || DEFAULT_OPTIONS.apiBaseUrl);
  const timeout = Number(input.requestTimeoutMs);
  const historyLimit = Number(input.historyLimit);
  const minDisplayedWidth = Number(input.minDisplayedWidth);
  const minDisplayedHeight = Number(input.minDisplayedHeight);
  const minNaturalWidth = Number(input.minNaturalWidth);
  const minNaturalHeight = Number(input.minNaturalHeight);
  const scanRootMarginPx = Number(input.scanRootMarginPx);
  const scanRequestConcurrency = Number(input.scanRequestConcurrency);
  const scanQueueLimit = Number(input.scanQueueLimit);
  const predictionCacheLimit = Number(input.predictionCacheLimit);
  const mockScenario = Object.values(MOCK_SCENARIOS).includes(input.mockScenario)
    ? input.mockScenario
    : DEFAULT_OPTIONS.mockScenario;
  return {
    apiBaseUrl: apiValidation.ok ? apiValidation.value : DEFAULT_OPTIONS.apiBaseUrl,
    requestTimeoutMs: Number.isFinite(timeout) ? clamp(Math.round(timeout), 3000, 120000) : DEFAULT_OPTIONS.requestTimeoutMs,
    autoOpenResults: Boolean(input.autoOpenResults ?? DEFAULT_OPTIONS.autoOpenResults),
    showModelScores: Boolean(input.showModelScores ?? DEFAULT_OPTIONS.showModelScores),
    historyLimit: Number.isFinite(historyLimit) ? clamp(Math.round(historyLimit), 1, 25) : DEFAULT_OPTIONS.historyLimit,
    minDisplayedWidth: Number.isFinite(minDisplayedWidth)
      ? clamp(Math.round(minDisplayedWidth), 32, 4096)
      : DEFAULT_OPTIONS.minDisplayedWidth,
    minDisplayedHeight: Number.isFinite(minDisplayedHeight)
      ? clamp(Math.round(minDisplayedHeight), 32, 4096)
      : DEFAULT_OPTIONS.minDisplayedHeight,
    minNaturalWidth: Number.isFinite(minNaturalWidth)
      ? clamp(Math.round(minNaturalWidth), 32, 10000)
      : DEFAULT_OPTIONS.minNaturalWidth,
    minNaturalHeight: Number.isFinite(minNaturalHeight)
      ? clamp(Math.round(minNaturalHeight), 32, 10000)
      : DEFAULT_OPTIONS.minNaturalHeight,
    scanRootMarginPx: Number.isFinite(scanRootMarginPx)
      ? clamp(Math.round(scanRootMarginPx), 0, 2000)
      : DEFAULT_OPTIONS.scanRootMarginPx,
    scanRequestConcurrency: Number.isFinite(scanRequestConcurrency)
      ? clamp(Math.round(scanRequestConcurrency), 1, 6)
      : DEFAULT_OPTIONS.scanRequestConcurrency,
    scanQueueLimit: Number.isFinite(scanQueueLimit)
      ? clamp(Math.round(scanQueueLimit), 1, 500)
      : DEFAULT_OPTIONS.scanQueueLimit,
    predictionCacheLimit: Number.isFinite(predictionCacheLimit)
      ? clamp(Math.round(predictionCacheLimit), 10, 2000)
      : DEFAULT_OPTIONS.predictionCacheLimit,
    overlaysEnabled: Boolean(input.overlaysEnabled ?? DEFAULT_OPTIONS.overlaysEnabled),
    diagnosticsEnabled: Boolean(input.diagnosticsEnabled ?? DEFAULT_OPTIONS.diagnosticsEnabled),
    mockModeEnabled: Boolean(input.mockModeEnabled ?? DEFAULT_OPTIONS.mockModeEnabled),
    mockScenario,
    globalScanningStopped: Boolean(input.globalScanningStopped ?? DEFAULT_OPTIONS.globalScanningStopped)
  };
}

export function validateSelectedImageUrl(value) {
  if (!value) {
    return { ok: false, code: "missing_image_url", message: "No selected image URL was provided." };
  }

  let url;
  try {
    url = new URL(value);
  } catch {
    return { ok: false, code: "invalid_image_url", message: "Selected image URL is invalid." };
  }

  if (url.protocol === "http:" || url.protocol === "https:") {
    return { ok: true, scheme: url.protocol, value: url.href };
  }

  if (url.protocol === "data:") {
    const mimeType = getDataUrlMimeType(value);
    if (!SUPPORTED_IMAGE_TYPES.includes(mimeType)) {
      return { ok: false, code: "unsupported_data_image_type", message: "Only JPEG, PNG, and WebP data images are supported." };
    }
    return { ok: true, scheme: "data:", value };
  }

  if (url.protocol === "blob:") {
    return {
      ok: false,
      code: "unsupported_blob_url",
      message: "Blob image URLs are not supported in this prototype. Try opening the original image URL."
    };
  }

  return {
    ok: false,
    code: "unsupported_image_url_scheme",
    message: "Only http, https, and supported data image URLs can be analysed."
  };
}

export function redactImageUrlForStorage(value) {
  if (!value) {
    return "";
  }
  try {
    const url = new URL(value);
    if (url.protocol === "data:") {
      return `data:${getDataUrlMimeType(value)};<redacted>`;
    }
    if (url.protocol === "blob:") {
      return `blob:${safeOriginFromBlobUrl(value)}/<redacted>`;
    }
    return url.href;
  } catch {
    return "[invalid image URL]";
  }
}

export function isImageLikeContentType(value) {
  const contentType = String(value || "").split(";")[0].trim().toLowerCase();
  return SUPPORTED_IMAGE_TYPES.includes(contentType);
}

export function validateImageBlob(blob) {
  if (!blob || typeof blob.size !== "number") {
    return { ok: false, code: "missing_image_blob", message: "Selected image could not be read." };
  }
  if (blob.size <= 0) {
    return { ok: false, code: "empty_image_blob", message: "Selected image is empty." };
  }
  if (blob.size > MAX_UPLOAD_BYTES) {
    return { ok: false, code: "image_too_large", message: "Selected image is larger than the 10 MB API limit." };
  }
  if (!isImageLikeContentType(blob.type)) {
    return { ok: false, code: "unsupported_image_blob_type", message: "Selected image type is not JPEG, PNG, or WebP." };
  }
  return { ok: true };
}

export function filenameFromUrl(value, contentType = "image/jpeg") {
  const extension = SUPPORTED_IMAGE_EXTENSIONS[String(contentType).split(";")[0].toLowerCase()] || ".jpg";
  try {
    const url = new URL(value);
    if (url.protocol === "data:") {
      return `selected-image${extension}`;
    }
    const name = decodeURIComponent(url.pathname.split("/").pop() || "");
    const sanitized = sanitizeFilename(name);
    if (sanitized) {
      return sanitized.includes(".") ? sanitized : `${sanitized}${extension}`;
    }
  } catch {
    // Fall through to generated filename.
  }
  return `selected-image-${Date.now()}${extension}`;
}

export function validateHealthResponse(payload) {
  if (!payload || typeof payload !== "object") {
    return { ok: false, code: "malformed_health_response", message: "Health response was not JSON." };
  }
  if (payload.status !== "ok" || payload.model_loaded !== true || payload.detector_loaded !== true) {
    return {
      ok: false,
      code: "api_not_ready",
      message: "Local detector is starting or not fully loaded."
    };
  }
  return { ok: true };
}

export function validatePredictionResponse(payload) {
  if (!payload || typeof payload !== "object") {
    return { ok: false, code: "malformed_prediction_response", message: "Prediction response was not JSON." };
  }

  if (!["completed", "no_face_detected"].includes(payload.status)) {
    return { ok: false, code: "unsupported_prediction_status", message: "Prediction response had an unsupported status." };
  }

  if (!payload.image || typeof payload.image.width !== "number" || typeof payload.image.height !== "number") {
    return { ok: false, code: "missing_image_info", message: "Prediction response is missing image information." };
  }

  if (!Array.isArray(payload.faces)) {
    return { ok: false, code: "missing_faces", message: "Prediction response is missing face results." };
  }

  const metadataCheck = validatePredictionMetadata(payload);
  if (!metadataCheck.ok) {
    return metadataCheck;
  }

  if (payload.status === "no_face_detected") {
    if (payload.faces_detected !== 0 || payload.faces.length !== 0) {
      return { ok: false, code: "inconsistent_no_face_response", message: "No-face response included face results." };
    }
    return { ok: true };
  }

  if (payload.faces_detected < 1 || payload.faces.length < 1) {
    return { ok: false, code: "completed_without_faces", message: "Completed response did not include any face results." };
  }

  for (const face of payload.faces) {
    const faceCheck = validateFace(face);
    if (!faceCheck.ok) {
      return faceCheck;
    }
  }

  return { ok: true };
}

export function validatePredictionMetadata(payload) {
  for (const key of ["model", "detector", "crop_strategy", "thresholds"]) {
    if (payload[key] == null) {
      continue;
    }
    if (typeof payload[key] !== "object") {
      return { ok: false, code: "malformed_prediction_metadata", message: "Prediction metadata is malformed." };
    }
    if (typeof payload[key].id !== "string" || typeof payload[key].version !== "string") {
      return { ok: false, code: "malformed_prediction_metadata", message: "Prediction metadata is missing an id or version." };
    }
  }
  return { ok: true };
}

export function validateFace(face) {
  if (!face || typeof face !== "object") {
    return { ok: false, code: "malformed_face_result", message: "Face result is malformed." };
  }
  if (!["Likely Fake", "Uncertain", "Likely Real"].includes(face.label)) {
    return { ok: false, code: "unsupported_face_label", message: "Face result label is unsupported." };
  }
  if (!isFiniteScore(face.real_score) || !isFiniteScore(face.fake_score)) {
    return { ok: false, code: "invalid_face_score", message: "Face result contains an invalid score." };
  }
  if (!face.bounding_box || !face.crop_box) {
    return { ok: false, code: "missing_face_box", message: "Face result is missing bounding box information." };
  }
  return { ok: true };
}

export function formatPercent(value) {
  if (!isFiniteScore(value)) {
    return "n/a";
  }
  return `${Math.round(Number(value) * 100)}%`;
}

export function formatScore(value) {
  if (!isFiniteScore(value)) {
    return "n/a";
  }
  return Number(value).toFixed(4);
}

export function summarizeFaces(faces = []) {
  const counts = { "Likely Fake": 0, Uncertain: 0, "Likely Real": 0 };
  for (const face of faces) {
    if (Object.prototype.hasOwnProperty.call(counts, face.label)) {
      counts[face.label] += 1;
    }
  }
  return counts;
}

export function buildResultSummary(response) {
  if (!response) {
    return "No analysis yet";
  }
  if (response.status === "no_face_detected") {
    return "No supported human face was detected.";
  }
  if (response.status === "completed") {
    const faces = Array.isArray(response.faces) ? response.faces : [];
    const counts = summarizeFaces(faces);
    return [
      `${faces.length} ${faces.length === 1 ? "face" : "faces"} analysed:`,
      `${counts["Likely Fake"]} Likely Fake`,
      `${counts.Uncertain} Uncertain`,
      `${counts["Likely Real"]} Likely Real`
    ].join(" ");
  }
  return "Analysis status unavailable";
}

export function mapApiError(status, body) {
  const structured = body && typeof body === "object" ? body.error : null;
  const code = structured?.code || `http_${status}`;
  const serverMessage = structured?.message;
  const fallbackMessages = {
    400: "The selected file is not a valid supported image.",
    413: "The selected image is too large for the local detector.",
    415: "The selected image format is not supported.",
    422: "The selected image could not be processed.",
    429: "The local detector is busy. Try again in a moment.",
    500: "The local detector failed while processing the image.",
    503: "The local detector is not ready.",
    504: "The local detector request timed out."
  };
  return {
    code,
    message: safeDisplayText(serverMessage || fallbackMessages[status] || "The local detector returned an error."),
    status
  };
}

export function safeDisplayText(value, maxLength = 300) {
  const text = String(value ?? "").replace(/[\u0000-\u001f\u007f]/g, " ").replace(/\s+/g, " ").trim();
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, Math.max(0, maxLength - 3))}...`;
}

export function truncateMiddle(value, maxLength = 80) {
  const text = safeDisplayText(value, maxLength * 2);
  if (text.length <= maxLength) {
    return text;
  }
  const side = Math.floor((maxLength - 3) / 2);
  return `${text.slice(0, side)}...${text.slice(text.length - side)}`;
}

export function originFromUrl(value) {
  try {
    const url = new URL(value);
    if (url.protocol === "data:") {
      return "data URL";
    }
    if (url.protocol === "blob:") {
      return safeOriginFromBlobUrl(value);
    }
    return url.origin;
  } catch {
    return "unknown";
  }
}

export function isSafeOriginalImageLink(value) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

export function getDataUrlMimeType(value) {
  const match = String(value).match(/^data:([^;,]+)[;,]/i);
  return match ? match[1].toLowerCase() : "text/plain";
}

function sanitizeFilename(value) {
  return String(value || "")
    .replace(/[\\/:*?"<>|]+/g, "-")
    .replace(/\s+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

function removeTrailingSlash(value) {
  return value.replace(/\/+$/, "");
}

function isLocalApiHost(hostname) {
  const host = String(hostname || "").toLowerCase();
  return host === "localhost" || host === "127.0.0.1" || host === "::1" || host === "[::1]";
}

function isFiniteScore(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 && number <= 1;
}

function safeOriginFromBlobUrl(value) {
  const withoutPrefix = String(value).replace(/^blob:/i, "");
  try {
    return new URL(withoutPrefix).origin;
  } catch {
    return "unknown";
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
