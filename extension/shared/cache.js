import { CACHE_STATUS, DEFAULT_OPTIONS, STORAGE_KEYS } from "./constants.js";
import { makeVersionSignature } from "./scanner.js";

export const CACHE_TTL_MS = {
  [CACHE_STATUS.completed]: 7 * 24 * 60 * 60 * 1000,
  [CACHE_STATUS.noFaceDetected]: 24 * 60 * 60 * 1000,
  [CACHE_STATUS.unsupported]: 24 * 60 * 60 * 1000,
  [CACHE_STATUS.temporaryFailure]: 5 * 60 * 1000
};

export function statusFromPredictionResponse(response) {
  if (response?.status === "completed") {
    return CACHE_STATUS.completed;
  }
  if (response?.status === "no_face_detected") {
    return CACHE_STATUS.noFaceDetected;
  }
  return CACHE_STATUS.temporaryFailure;
}

export function createCacheEntry({
  sourceKey,
  contentKey = "",
  imageUrl = "",
  status,
  response = null,
  error = null,
  metadata,
  contentHash = "",
  now = Date.now()
}) {
  const safeStatus = Object.values(CACHE_STATUS).includes(status) ? status : CACHE_STATUS.temporaryFailure;
  return {
    sourceKey: String(sourceKey || ""),
    contentKey: String(contentKey || ""),
    imageUrl: String(imageUrl || ""),
    status: safeStatus,
    response,
    error,
    contentHash: String(contentHash || ""),
    versionSignature: makeVersionSignature(metadata),
    createdAt: now,
    expiresAt: now + ttlForStatus(safeStatus)
  };
}

export function isCacheEntryFresh(entry, metadata, now = Date.now()) {
  if (!entry || typeof entry !== "object") {
    return false;
  }
  if (!entry.expiresAt || Number(entry.expiresAt) <= now) {
    return false;
  }
  return entry.versionSignature === makeVersionSignature(metadata);
}

export function findFreshCacheEntry(cache, key, metadata, now = Date.now()) {
  const entry = cache && typeof cache === "object" ? cache[key] : null;
  return isCacheEntryFresh(entry, metadata, now) ? entry : null;
}

export function prunePredictionCache(cache, limit = DEFAULT_OPTIONS.predictionCacheLimit, now = Date.now()) {
  const maxEntries = Math.max(10, Math.min(2000, Number(limit) || DEFAULT_OPTIONS.predictionCacheLimit));
  const entries = Object.entries(cache || {})
    .filter(([, entry]) => entry && Number(entry.expiresAt || 0) > now)
    .sort((a, b) => Number(b[1].createdAt || 0) - Number(a[1].createdAt || 0));
  return Object.fromEntries(entries.slice(0, maxEntries));
}

export async function getPredictionCache() {
  const stored = await storageGet(STORAGE_KEYS.predictionCache);
  return stored && typeof stored === "object" ? stored : {};
}

export async function putPredictionCacheEntry(entry, limit = DEFAULT_OPTIONS.predictionCacheLimit) {
  const cache = await getPredictionCache();
  if (entry.sourceKey) {
    cache[entry.sourceKey] = entry;
  }
  if (entry.contentKey) {
    cache[entry.contentKey] = entry;
  }
  const pruned = prunePredictionCache(cache, limit);
  await storageSet({ [STORAGE_KEYS.predictionCache]: pruned });
  return entry;
}

export async function clearPredictionCache() {
  await storageSet({ [STORAGE_KEYS.predictionCache]: {} });
}

function ttlForStatus(status) {
  return CACHE_TTL_MS[status] || CACHE_TTL_MS[CACHE_STATUS.temporaryFailure];
}

function storageGet(key) {
  return new Promise((resolve) => {
    chrome.storage.local.get(key, (items) => {
      resolve(items[key]);
    });
  });
}

function storageSet(items) {
  return new Promise((resolve) => {
    chrome.storage.local.set(items, resolve);
  });
}
