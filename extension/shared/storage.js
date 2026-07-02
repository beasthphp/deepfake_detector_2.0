import { DEFAULT_OPTIONS, SITE_MODE, STORAGE_KEYS } from "./constants.js";
import { getSiteMode, setSiteMode } from "./scanner.js";
import { normalizeOptions, safeDisplayText } from "./validation.js";

export async function getOptions() {
  const stored = await storageGet(STORAGE_KEYS.options);
  return normalizeOptions({ ...DEFAULT_OPTIONS, ...(stored || {}) });
}

export async function saveOptions(options) {
  const normalized = normalizeOptions(options);
  await storageSet({ [STORAGE_KEYS.options]: normalized });
  return normalized;
}

export async function getCurrentRequest() {
  return (await storageGet(STORAGE_KEYS.currentRequest)) || null;
}

export async function getHistory() {
  return (await storageGet(STORAGE_KEYS.history)) || [];
}

export async function getSiteSettings() {
  const stored = await storageGet(STORAGE_KEYS.siteSettings);
  return stored && typeof stored === "object" ? stored : {};
}

export async function saveSiteSettings(settings) {
  const safeSettings = {};
  for (const [origin, mode] of Object.entries(settings || {})) {
    safeSettings[safeDisplayText(origin, 300)] = Object.values(SITE_MODE).includes(mode) ? mode : SITE_MODE.manualOnly;
  }
  await storageSet({ [STORAGE_KEYS.siteSettings]: safeSettings });
  return safeSettings;
}

export async function getSiteModeForUrl(pageUrl) {
  return getSiteMode(await getSiteSettings(), pageUrl);
}

export async function saveSiteModeForUrl(pageUrl, mode) {
  const next = setSiteMode(await getSiteSettings(), pageUrl, mode);
  return saveSiteSettings(next);
}

export async function setGlobalScanningStopped(stopped) {
  const options = await getOptions();
  return saveOptions({ ...options, globalScanningStopped: Boolean(stopped) });
}

export async function saveAnalysisRecord(record, historyLimit) {
  const safeRecord = sanitizeRecord(record);
  const history = await getHistory();
  const nextHistory = upsertHistory(history, safeRecord, historyLimit);
  await storageSet({
    [STORAGE_KEYS.currentRequest]: safeRecord,
    [STORAGE_KEYS.history]: nextHistory
  });
  return safeRecord;
}

export async function clearStoredResults() {
  await storageSet({
    [STORAGE_KEYS.currentRequest]: null,
    [STORAGE_KEYS.history]: []
  });
}

export function upsertHistory(history, record, limit = DEFAULT_OPTIONS.historyLimit) {
  const records = Array.isArray(history) ? history : [];
  const withoutExisting = records.filter((item) => item.analysisId !== record.analysisId);
  return trimHistory([record, ...withoutExisting], limit);
}

export function trimHistory(history, limit = DEFAULT_OPTIONS.historyLimit) {
  const records = Array.isArray(history) ? history : [];
  const safeLimit = Math.max(1, Math.min(25, Number(limit) || DEFAULT_OPTIONS.historyLimit));
  return records.slice(0, safeLimit);
}

export function sanitizeRecord(record) {
  return {
    analysisId: safeDisplayText(record.analysisId || cryptoRandomId()),
    status: safeDisplayText(record.status || "idle"),
    stage: safeDisplayText(record.stage || ""),
    stageMessage: safeDisplayText(record.stageMessage || ""),
    selectedImageUrl: safeDisplayText(record.selectedImageUrl || "", 600),
    pageUrl: safeDisplayText(record.pageUrl || "", 600),
    requestId: safeDisplayText(record.requestId || ""),
    apiBaseUrl: safeDisplayText(record.apiBaseUrl || ""),
    response: record.response || null,
    error: record.error ? sanitizeError(record.error) : null,
    timestamp: safeDisplayText(record.timestamp || new Date().toISOString())
  };
}

export function sanitizeError(error) {
  return {
    code: safeDisplayText(error.code || "extension_error"),
    message: safeDisplayText(error.message || "Analysis failed."),
    status: typeof error.status === "number" ? error.status : null,
    retryable: Boolean(error.retryable)
  };
}

export function storageGet(key) {
  return new Promise((resolve) => {
    chrome.storage.local.get(key, (items) => {
      resolve(items[key]);
    });
  });
}

export function storageSet(items) {
  return new Promise((resolve) => {
    chrome.storage.local.set(items, resolve);
  });
}

function cryptoRandomId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `analysis-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
