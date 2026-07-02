import { DEFAULT_BACKEND_METADATA, DEFAULT_OPTIONS, SITE_MODE } from "./constants.js";

export const SCANNER_CACHE_SCHEMA_VERSION = "phase5-visible-image-scan-v1";

const HTTP_PROTOCOLS = new Set(["http:", "https:"]);
const INTERNAL_PAGE_PROTOCOLS = new Set(["chrome:", "edge:", "about:", "devtools:", "chrome-extension:", "moz-extension:"]);
const ICON_NAME_PATTERN = /(^|[-_/])(favicon|icon|logo|sprite|badge|tracking|pixel)([-_.?/]|$)/i;

export function isSupportedPageUrl(value) {
  let url;
  try {
    url = new URL(value || "");
  } catch {
    return false;
  }
  if (INTERNAL_PAGE_PROTOCOLS.has(url.protocol)) {
    return false;
  }
  return HTTP_PROTOCOLS.has(url.protocol);
}

export function originKeyFromUrl(value) {
  try {
    const url = new URL(value || "");
    return HTTP_PROTOCOLS.has(url.protocol) ? url.origin : "";
  } catch {
    return "";
  }
}

export function normalizeSiteMode(value) {
  return Object.values(SITE_MODE).includes(value) ? value : SITE_MODE.manualOnly;
}

export function getSiteMode(siteSettings, pageUrl) {
  const origin = originKeyFromUrl(pageUrl);
  if (!origin || !siteSettings || typeof siteSettings !== "object") {
    return SITE_MODE.manualOnly;
  }
  return normalizeSiteMode(siteSettings[origin]);
}

export function setSiteMode(siteSettings, pageUrl, mode) {
  const origin = originKeyFromUrl(pageUrl);
  if (!origin) {
    return { ...(siteSettings || {}) };
  }
  return {
    ...(siteSettings || {}),
    [origin]: normalizeSiteMode(mode)
  };
}

export function isScannableImageUrl(value) {
  let url;
  try {
    url = new URL(value || "");
  } catch {
    return false;
  }
  return HTTP_PROTOCOLS.has(url.protocol);
}

export function normalizeBackendMetadata(input = {}) {
  const cropStrategy = input.cropStrategy || input.crop_strategy || {};
  return {
    model: normalizeComponent(input.model, DEFAULT_BACKEND_METADATA.model),
    detector: normalizeComponent(input.detector, DEFAULT_BACKEND_METADATA.detector),
    cropStrategy: normalizeComponent(cropStrategy, DEFAULT_BACKEND_METADATA.cropStrategy),
    thresholds: normalizeComponent(input.thresholds, DEFAULT_BACKEND_METADATA.thresholds)
  };
}

export function backendMetadataFromHealth(payload = {}) {
  return normalizeBackendMetadata({
    model: {
      id: payload.model_id || DEFAULT_BACKEND_METADATA.model.id,
      version: payload.model_version || DEFAULT_BACKEND_METADATA.model.version
    },
    detector: {
      id: payload.detector_id || DEFAULT_BACKEND_METADATA.detector.id,
      version: payload.detector_version || DEFAULT_BACKEND_METADATA.detector.version
    },
    cropStrategy: {
      id: payload.crop_strategy_id || DEFAULT_BACKEND_METADATA.cropStrategy.id,
      version: payload.crop_strategy_version || DEFAULT_BACKEND_METADATA.cropStrategy.version
    },
    thresholds: {
      id: payload.threshold_id || DEFAULT_BACKEND_METADATA.thresholds.id,
      version: payload.threshold_version || DEFAULT_BACKEND_METADATA.thresholds.version
    }
  });
}

export function backendMetadataFromPrediction(payload = {}) {
  return normalizeBackendMetadata({
    model: payload.model,
    detector: payload.detector,
    cropStrategy: payload.crop_strategy,
    thresholds: payload.thresholds
  });
}

export function makeVersionSignature(metadata = DEFAULT_BACKEND_METADATA) {
  const normalized = normalizeBackendMetadata(metadata);
  return [
    SCANNER_CACHE_SCHEMA_VERSION,
    `model:${normalized.model.id}@${normalized.model.version}`,
    `detector:${normalized.detector.id}@${normalized.detector.version}`,
    `crop:${normalized.cropStrategy.id}@${normalized.cropStrategy.version}`,
    `thresholds:${normalized.thresholds.id}@${normalized.thresholds.version}`
  ].join("|");
}

export function buildImageSourceKey(descriptor, metadata = DEFAULT_BACKEND_METADATA) {
  const sourceUrl = normalizeImageSourceUrl(descriptor?.currentSrc || descriptor?.src || descriptor?.url || "");
  const naturalWidth = positiveInteger(descriptor?.naturalWidth);
  const naturalHeight = positiveInteger(descriptor?.naturalHeight);
  return stableKey({
    kind: "source",
    sourceUrl,
    naturalWidth,
    naturalHeight,
    versionSignature: makeVersionSignature(metadata)
  });
}

export function buildImageContentKey(descriptor, metadata = DEFAULT_BACKEND_METADATA, contentHash = "") {
  return stableKey({
    kind: "content",
    sourceKey: buildImageSourceKey(descriptor, metadata),
    contentHash: String(contentHash || "unhashed")
  });
}

export function evaluateImageEligibility(descriptor, options = DEFAULT_OPTIONS, processedSourceKeys = new Set(), metadata = DEFAULT_BACKEND_METADATA) {
  const merged = { ...DEFAULT_OPTIONS, ...(options || {}) };
  const sourceUrl = descriptor?.currentSrc || descriptor?.src || descriptor?.url || "";
  if (!isScannableImageUrl(sourceUrl)) {
    return skipped("unsupported_url");
  }
  if (!descriptor?.isLoaded) {
    return skipped("not_loaded");
  }
  if (descriptor?.isVisible === false || descriptor?.visibilityState === "hidden") {
    return skipped("hidden");
  }

  const displayedWidth = positiveNumber(descriptor?.displayedWidth);
  const displayedHeight = positiveNumber(descriptor?.displayedHeight);
  const naturalWidth = positiveNumber(descriptor?.naturalWidth);
  const naturalHeight = positiveNumber(descriptor?.naturalHeight);

  if (displayedWidth <= 1 || displayedHeight <= 1 || naturalWidth <= 1 || naturalHeight <= 1) {
    return skipped("tracking_pixel");
  }
  if (displayedWidth < merged.minDisplayedWidth || displayedHeight < merged.minDisplayedHeight) {
    return skipped("display_too_small");
  }
  if (naturalWidth < merged.minNaturalWidth || naturalHeight < merged.minNaturalHeight) {
    return skipped("natural_too_small");
  }
  if (hasExtremeAspectRatio(displayedWidth, displayedHeight) || hasExtremeAspectRatio(naturalWidth, naturalHeight)) {
    return skipped("extreme_aspect_ratio");
  }
  if (isLikelyIconOrLogo(sourceUrl, naturalWidth, naturalHeight, displayedWidth, displayedHeight)) {
    return skipped("likely_icon_or_logo");
  }

  const sourceKey = buildImageSourceKey(descriptor, metadata);
  if (processedSourceKeys.has(sourceKey)) {
    return skipped("already_processed", sourceKey);
  }

  return { eligible: true, reason: "eligible", sourceKey };
}

export function isStaleScanResult(result, current) {
  if (!result || !current) {
    return true;
  }
  return (
    String(result.sessionId || "") !== String(current.sessionId || "") ||
    String(result.imageId || "") !== String(current.imageId || "") ||
    String(result.sourceKey || "") !== String(current.sourceKey || "")
  );
}

export function normalizeImageSourceUrl(value) {
  try {
    return new URL(value || "").href;
  } catch {
    return "";
  }
}

function normalizeComponent(value, fallback) {
  const input = value && typeof value === "object" ? value : {};
  return {
    id: nonEmptyString(input.id, fallback.id),
    version: nonEmptyString(input.version, fallback.version)
  };
}

function nonEmptyString(value, fallback) {
  const text = String(value || "").trim();
  return text || fallback;
}

function skipped(reason, sourceKey = "") {
  return { eligible: false, reason, sourceKey };
}

function stableKey(value) {
  return JSON.stringify(value);
}

function positiveInteger(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.round(number) : 0;
}

function positiveNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : 0;
}

function hasExtremeAspectRatio(width, height) {
  const ratio = width / height;
  return ratio > 4 || ratio < 0.25;
}

function isLikelyIconOrLogo(urlValue, naturalWidth, naturalHeight, displayedWidth, displayedHeight) {
  if (!ICON_NAME_PATTERN.test(urlValue)) {
    return false;
  }
  const compactNatural = Math.max(naturalWidth, naturalHeight) <= 512;
  const compactDisplay = Math.max(displayedWidth, displayedHeight) <= 256;
  return compactNatural || compactDisplay;
}
