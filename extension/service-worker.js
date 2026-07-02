import {
  CACHE_STATUS,
  IMAGE_SCAN_STATE,
  MENU_ID,
  REQUEST_STAGE,
  REQUEST_STATUS,
  SCAN_COMMAND,
  SITE_MODE,
  STAGE_MESSAGES
} from "./shared/constants.js";
import { checkHealth, ExtensionRequestError, fetchSelectedImage, predictImage } from "./shared/api-client.js";
import {
  clearPredictionCache,
  createCacheEntry,
  findFreshCacheEntry,
  getPredictionCache,
  putPredictionCacheEntry,
  statusFromPredictionResponse
} from "./shared/cache.js";
import { predictMockImage, mockHealthPayload } from "./shared/mock-api.js";
import { BoundedRequestQueue } from "./shared/request-queue.js";
import {
  backendMetadataFromHealth,
  buildImageContentKey,
  buildImageSourceKey,
  isSupportedPageUrl,
  originKeyFromUrl
} from "./shared/scanner.js";
import {
  getOptions,
  getSiteModeForUrl,
  saveAnalysisRecord,
  saveSiteModeForUrl,
  setGlobalScanningStopped
} from "./shared/storage.js";
import { redactImageUrlForStorage, safeDisplayText, validatePredictionResponse } from "./shared/validation.js";

const sessionsByTab = new Map();
const subscribersBySourceKey = new Map();
const pendingSourceKeys = new Set();

const scanQueue = new BoundedRequestQueue({
  worker: processQueuedScanItem,
  onError: (error, item) => {
    console.error("Deepfake scan queue error", sanitizeErrorForLog(error), item?.sourceKey || "");
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: MENU_ID,
      title: "Analyze face for deepfake",
      contexts: ["image"]
    });
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== MENU_ID) {
    return;
  }
  analyzeSelectedImage(info, tab).catch((error) => {
    console.error("Deepfake analysis failed", sanitizeErrorForLog(error));
  });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || typeof message.type !== "string") {
    return false;
  }
  handleRuntimeMessage(message, sender)
    .then((response) => sendResponse(response))
    .catch((error) => {
      sendResponse({ ok: false, error: errorToRecord(error) });
    });
  return true;
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete" || !tab?.url) {
    return;
  }
  maybeStartEnabledSiteScan({ id: tabId, url: tab.url }).catch((error) => {
    console.warn("Could not start enabled-site scan", sanitizeErrorForLog(error));
  });
});

chrome.tabs.onRemoved.addListener((tabId) => {
  stopScanForTab(tabId, "tab_removed").catch(() => {});
});

async function analyzeSelectedImage(info, tab) {
  const options = await getOptions();
  const analysisId = crypto.randomUUID();
  const originalImageUrl = info.srcUrl || "";
  const frameId = typeof info.frameId === "number" ? info.frameId : undefined;
  const baseRecord = {
    analysisId,
    status: REQUEST_STATUS.running,
    stage: REQUEST_STAGE.queued,
    stageMessage: STAGE_MESSAGES[REQUEST_STAGE.queued],
    selectedImageUrl: redactImageUrlForStorage(originalImageUrl),
    pageUrl: info.pageUrl || tab?.url || "",
    requestId: "",
    apiBaseUrl: options.apiBaseUrl,
    response: null,
    error: null,
    timestamp: new Date().toISOString()
  };

  await saveAnalysisRecord(baseRecord, options.historyLimit);
  if (options.autoOpenResults) {
    chrome.tabs.create({ url: chrome.runtime.getURL("results/results.html") });
  }
  await notifyPage(tab?.id, frameId, "Deepfake analysis started...", "running");

  try {
    await updateRecord(baseRecord, options, REQUEST_STAGE.checkingApi, STAGE_MESSAGES[REQUEST_STAGE.checkingApi]);
    if (!options.mockModeEnabled) {
      await checkHealth(options.apiBaseUrl, options.requestTimeoutMs);
    }

    await updateRecord(baseRecord, options, REQUEST_STAGE.downloadingImage, STAGE_MESSAGES[REQUEST_STAGE.downloadingImage]);
    const image = await fetchSelectedImage(originalImageUrl, options.requestTimeoutMs);

    await updateRecord(baseRecord, options, REQUEST_STAGE.detectingFaces, STAGE_MESSAGES[REQUEST_STAGE.detectingFaces]);
    await updateRecord(baseRecord, options, REQUEST_STAGE.analyzingFaces, STAGE_MESSAGES[REQUEST_STAGE.analyzingFaces]);
    const response = options.mockModeEnabled
      ? await predictMockImage({ scenario: options.mockScenario, imageBlob: image.blob, timeoutMs: options.requestTimeoutMs })
      : await predictImage(options.apiBaseUrl, image.blob, image.filename, options.requestTimeoutMs);

    const completedStatus =
      response.status === "no_face_detected" ? REQUEST_STATUS.noFaceDetected : REQUEST_STATUS.completed;
    const completedRecord = {
      ...baseRecord,
      status: completedStatus,
      stage: REQUEST_STAGE.completed,
      stageMessage: STAGE_MESSAGES[REQUEST_STAGE.completed],
      requestId: response.request_id || "",
      response,
      timestamp: new Date().toISOString()
    };
    await saveAnalysisRecord(completedRecord, options.historyLimit);

    if (response.status === "no_face_detected") {
      await notifyPage(tab?.id, frameId, "No face detected", "no_face");
    } else {
      await notifyPage(tab?.id, frameId, "Analysis completed - open extension results", "completed");
    }
  } catch (error) {
    const safeError = errorToRecord(error);
    await saveAnalysisRecord(
      {
        ...baseRecord,
        status: REQUEST_STATUS.failed,
        stage: REQUEST_STAGE.failed,
        stageMessage: STAGE_MESSAGES[REQUEST_STAGE.failed],
        error: safeError,
        timestamp: new Date().toISOString()
      },
      options.historyLimit
    );
    await notifyPage(tab?.id, frameId, notificationForError(safeError), "failed");
  }
}

async function handleRuntimeMessage(message, sender) {
  if (message.type === "DFD_SCAN_COMMAND") {
    return handleScanCommand(message);
  }
  if (message.type === "DFD_SCAN_IMAGE_DISCOVERED") {
    return handleDiscoveredImage(message, sender);
  }
  if (message.type === "DFD_SCAN_IMAGE_REMOVED") {
    return handleRemovedImage(message, sender);
  }
  return { ok: true, ignored: true };
}

async function handleScanCommand(message) {
  const tab = await resolveCommandTab(message);
  const command = message.command;

  if (command === SCAN_COMMAND.getState) {
    return scanStateForTab(tab);
  }
  if (command === SCAN_COMMAND.clearCache) {
    await clearPredictionCache();
    return { ok: true, message: "Prediction cache cleared." };
  }
  if (command === SCAN_COMMAND.globalStop) {
    await setGlobalScanningStopped(true);
    await stopAllScans("global_stop");
    return { ok: true, message: "Global scanning stopped." };
  }
  if (command === SCAN_COMMAND.resumeGlobal) {
    await setGlobalScanningStopped(false);
    return { ok: true, message: "Global scanning resumed." };
  }

  if (!tab?.id || !isSupportedPageUrl(tab.url)) {
    return { ok: false, error: { code: "unsupported_page", message: "This page cannot be scanned." } };
  }

  if (command === SCAN_COMMAND.stop) {
    await stopScanForTab(tab.id, "user_stop");
    return { ok: true, message: "Scanning stopped." };
  }
  if (command === SCAN_COMMAND.clearPage) {
    await sendToTab(tab.id, { type: "DFD_SCAN_CLEAR" });
    return { ok: true, message: "Page results cleared." };
  }
  if (command === SCAN_COMMAND.enableSite) {
    await saveSiteModeForUrl(tab.url, SITE_MODE.enabled);
    return startScanForTab(tab, { clear: true, reason: "site_enabled" });
  }
  if (command === SCAN_COMMAND.disableSite) {
    await saveSiteModeForUrl(tab.url, SITE_MODE.disabled);
    await stopScanForTab(tab.id, "site_disabled");
    return { ok: true, message: "Scanning disabled on this site." };
  }
  if (command === SCAN_COMMAND.rescan) {
    await stopScanForTab(tab.id, "rescan");
    return startScanForTab(tab, { clear: true, reason: "rescan" });
  }
  if (command === SCAN_COMMAND.start) {
    return startScanForTab(tab, { clear: false, reason: "manual_scan" });
  }
  return { ok: false, error: { code: "unknown_scan_command", message: "Unknown scan command." } };
}

async function startScanForTab(tab, { clear = false, reason = "manual_scan" } = {}) {
  const options = await getOptions();
  if (options.globalScanningStopped) {
    return { ok: false, error: { code: "global_stop", message: "Global scanning is stopped." } };
  }
  if (!isSupportedPageUrl(tab.url)) {
    return { ok: false, error: { code: "unsupported_page", message: "This page cannot be scanned." } };
  }

  const siteMode = await getSiteModeForUrl(tab.url);
  if (siteMode === SITE_MODE.disabled) {
    return { ok: false, error: { code: "site_disabled", message: "Scanning is disabled on this site." } };
  }

  let backendMetadata;
  try {
    backendMetadata = await loadBackendMetadata(options);
  } catch (error) {
    await ensureContentScript(tab.id);
    await sendToTab(tab.id, {
      type: "DFD_SCAN_TOAST",
      message: notificationForError(errorToRecord(error)),
      state: "failed"
    });
    return { ok: false, error: errorToRecord(error) };
  }

  scanQueue.configure({
    concurrency: options.scanRequestConcurrency,
    maxQueued: options.scanQueueLimit
  });
  await ensureContentScript(tab.id);
  if (clear) {
    await sendToTab(tab.id, { type: "DFD_SCAN_CLEAR" });
  }

  const session = {
    tabId: tab.id,
    pageUrl: tab.url,
    origin: originKeyFromUrl(tab.url),
    sessionId: crypto.randomUUID(),
    reason,
    active: true,
    options,
    backendMetadata,
    startedAt: Date.now()
  };
  sessionsByTab.set(tab.id, session);
  await sendToTab(tab.id, {
    type: "DFD_SCAN_START",
    sessionId: session.sessionId,
    options: publicScannerOptions(options),
    siteMode,
    backendMetadata
  });
  return {
    ok: true,
    message: options.mockModeEnabled ? "Mock page scanning started." : "Page scanning started.",
    sessionId: session.sessionId,
    siteMode,
    queue: scanQueue.stats()
  };
}

async function maybeStartEnabledSiteScan(tab) {
  const options = await getOptions();
  if (options.globalScanningStopped || !isSupportedPageUrl(tab.url)) {
    return;
  }
  const mode = await getSiteModeForUrl(tab.url);
  if (mode === SITE_MODE.enabled) {
    await startScanForTab(tab, { clear: true, reason: "enabled_site_navigation" });
  }
}

async function stopScanForTab(tabId, reason) {
  const session = sessionsByTab.get(tabId);
  if (session) {
    session.active = false;
  }
  scanQueue.cancelWhere((item) => item.tabId === tabId);
  pendingSourceKeys.forEach((sourceKey) => {
    const subscribers = subscribersBySourceKey.get(sourceKey);
    if (!subscribers) {
      return;
    }
    for (const subscriber of subscribers.values()) {
      if (subscriber.tabId === tabId) {
        subscribers.delete(subscriberKey(subscriber));
      }
    }
    if (subscribers.size === 0) {
      subscribersBySourceKey.delete(sourceKey);
      pendingSourceKeys.delete(sourceKey);
    }
  });
  sessionsByTab.delete(tabId);
  await sendToTab(tabId, { type: "DFD_SCAN_STOP", reason });
}

async function stopAllScans(reason) {
  const tabIds = Array.from(sessionsByTab.keys());
  await Promise.all(tabIds.map((tabId) => stopScanForTab(tabId, reason)));
}

async function handleDiscoveredImage(message, sender) {
  const tabId = sender.tab?.id;
  const session = tabId == null ? null : sessionsByTab.get(tabId);
  if (!session?.active || message.sessionId !== session.sessionId) {
    return { ok: false, ignored: true, reason: "stale_session" };
  }

  const descriptor = message.image || {};
  const sourceKey = buildImageSourceKey(descriptor, session.backendMetadata);
  const subscriber = {
    tabId,
    imageId: descriptor.id,
    sessionId: session.sessionId,
    sourceKey,
    descriptor
  };

  const sourceCache = findFreshCacheEntry(await getPredictionCache(), sourceKey, session.backendMetadata);
  if (sourceCache) {
    await deliverCacheEntry(sourceCache, subscriber);
    return { ok: true, cacheHit: true };
  }

  addSubscriber(sourceKey, subscriber);
  if (pendingSourceKeys.has(sourceKey)) {
    await sendImageState(subscriber, IMAGE_SCAN_STATE.queued, "Queued behind matching image.");
    return { ok: true, deduplicated: true };
  }

  const accepted = scanQueue.enqueue({
    id: `${sourceKey}:${Date.now()}`,
    tabId,
    sessionId: session.sessionId,
    sourceKey,
    descriptor,
    retryCount: 0,
    priority: Number(message.priority || 0)
  });
  if (!accepted.accepted) {
    removeSubscriber(sourceKey, subscriber);
    await sendImageError(subscriber, {
      code: "scan_queue_full",
      message: "The scan queue is full.",
      retryable: true
    });
    return { ok: false, error: { code: "scan_queue_full", message: "The scan queue is full." } };
  }

  pendingSourceKeys.add(sourceKey);
  await sendImageState(subscriber, IMAGE_SCAN_STATE.queued, "Queued for analysis.");
  return { ok: true, queued: true, queue: scanQueue.stats() };
}

async function handleRemovedImage(message, sender) {
  const tabId = sender.tab?.id;
  if (tabId == null) {
    return { ok: true };
  }
  const sourceKey = String(message.sourceKey || "");
  const imageId = String(message.imageId || "");
  const subscribers = subscribersBySourceKey.get(sourceKey);
  if (subscribers) {
    subscribers.delete(`${tabId}:${imageId}`);
    if (subscribers.size === 0) {
      subscribersBySourceKey.delete(sourceKey);
      pendingSourceKeys.delete(sourceKey);
      scanQueue.cancelWhere((item) => item.sourceKey === sourceKey);
    }
  }
  return { ok: true };
}

async function processQueuedScanItem(item, queueContext) {
  let requeued = false;
  const session = sessionsByTab.get(item.tabId);
  if (!session?.active || item.sessionId !== session.sessionId || queueContext.isCanceled()) {
    cleanupSource(item.sourceKey);
    return;
  }

  try {
    await sendToSourceSubscribers(item.sourceKey, IMAGE_SCAN_STATE.downloading, "Downloading image.");
    const image = await fetchSelectedImage(item.descriptor.currentSrc || item.descriptor.src, session.options.requestTimeoutMs);
    if (queueContext.isCanceled()) {
      cleanupSource(item.sourceKey);
      return;
    }

    const contentHash = await hashBlob(image.blob);
    const contentKey = buildImageContentKey(item.descriptor, session.backendMetadata, contentHash);
    const cachedByContent = findFreshCacheEntry(await getPredictionCache(), contentKey, session.backendMetadata);
    if (cachedByContent) {
      await deliverCacheEntryToSource(cachedByContent, item.sourceKey);
      cleanupSource(item.sourceKey);
      return;
    }

    await sendToSourceSubscribers(item.sourceKey, IMAGE_SCAN_STATE.analysing, "Analysing detected faces.");
    const response = session.options.mockModeEnabled
      ? await predictMockImage({
          scenario: session.options.mockScenario,
          imageBlob: image.blob,
          descriptor: item.descriptor,
          timeoutMs: session.options.requestTimeoutMs
        })
      : await predictImage(session.options.apiBaseUrl, image.blob, image.filename, session.options.requestTimeoutMs);

    const validation = validatePredictionResponse(response);
    if (!validation.ok) {
      throw new ExtensionRequestError(validation.code, validation.message, { retryable: false });
    }

    const entry = createCacheEntry({
      sourceKey: item.sourceKey,
      contentKey,
      imageUrl: item.descriptor.currentSrc || item.descriptor.src,
      status: statusFromPredictionResponse(response),
      response,
      metadata: session.backendMetadata,
      contentHash
    });
    await putPredictionCacheEntry(entry, session.options.predictionCacheLimit);
    await deliverPredictionToSource(response, item.sourceKey, false);
  } catch (error) {
    if (shouldRetryScanError(error) && item.retryCount < 1 && !queueContext.isCanceled()) {
      requeued = true;
      await sendToSourceSubscribers(item.sourceKey, IMAGE_SCAN_STATE.apiOffline, "Detector unavailable. Retrying once.");
      setTimeout(() => {
        const accepted = scanQueue.enqueue({
          ...item,
          id: `${item.sourceKey}:retry:${Date.now()}`,
          retryCount: item.retryCount + 1
        });
        if (!accepted.accepted) {
          sendErrorToSource(item.sourceKey, {
            code: "scan_queue_full",
            message: "The scan queue is full.",
            retryable: true
          }).finally(() => cleanupSource(item.sourceKey));
        }
      }, 1200);
      return;
    }

    const safeError = errorToRecord(error);
    const status = safeError.retryable ? CACHE_STATUS.temporaryFailure : CACHE_STATUS.unsupported;
    const entry = createCacheEntry({
      sourceKey: item.sourceKey,
      imageUrl: item.descriptor.currentSrc || item.descriptor.src,
      status,
      error: safeError,
      metadata: session.backendMetadata
    });
    await putPredictionCacheEntry(entry, session.options.predictionCacheLimit);
    await sendErrorToSource(item.sourceKey, safeError);
  } finally {
    if (!requeued) {
      cleanupSource(item.sourceKey);
    }
  }
}

async function deliverCacheEntry(entry, subscriber) {
  if (entry.response) {
    await sendImageResult(subscriber, entry.response, true);
  } else {
    await sendImageError(subscriber, entry.error || { code: "cached_failure", message: "Cached scan failure." });
  }
}

async function deliverCacheEntryToSource(entry, sourceKey) {
  if (entry.response) {
    await deliverPredictionToSource(entry.response, sourceKey, true);
  } else {
    await sendErrorToSource(sourceKey, entry.error || { code: "cached_failure", message: "Cached scan failure." });
  }
}

async function deliverPredictionToSource(response, sourceKey, cacheHit) {
  const state = response.status === "no_face_detected" ? IMAGE_SCAN_STATE.noFaceDetected : IMAGE_SCAN_STATE.completed;
  const subscribers = subscribersBySourceKey.get(sourceKey);
  if (!subscribers) {
    return;
  }
  await Promise.all(
    Array.from(subscribers.values()).map((subscriber) => sendImageResult(subscriber, response, cacheHit, state))
  );
}

async function sendToSourceSubscribers(sourceKey, state, message) {
  const subscribers = subscribersBySourceKey.get(sourceKey);
  if (!subscribers) {
    return;
  }
  await Promise.all(Array.from(subscribers.values()).map((subscriber) => sendImageState(subscriber, state, message)));
}

async function sendErrorToSource(sourceKey, error) {
  const subscribers = subscribersBySourceKey.get(sourceKey);
  if (!subscribers) {
    return;
  }
  await Promise.all(Array.from(subscribers.values()).map((subscriber) => sendImageError(subscriber, error)));
}

async function sendImageState(subscriber, state, message) {
  await sendToTab(subscriber.tabId, {
    type: "DFD_SCAN_IMAGE_STATE",
    sessionId: subscriber.sessionId,
    imageId: subscriber.imageId,
    sourceKey: subscriber.sourceKey,
    localSignature: subscriber.descriptor.localSignature || "",
    state,
    message
  });
}

async function sendImageResult(subscriber, response, cacheHit, state = IMAGE_SCAN_STATE.completed) {
  await sendToTab(subscriber.tabId, {
    type: "DFD_SCAN_IMAGE_RESULT",
    sessionId: subscriber.sessionId,
    imageId: subscriber.imageId,
    sourceKey: subscriber.sourceKey,
    localSignature: subscriber.descriptor.localSignature || "",
    state,
    response,
    cacheHit
  });
}

async function sendImageError(subscriber, error) {
  await sendToTab(subscriber.tabId, {
    type: "DFD_SCAN_IMAGE_ERROR",
    sessionId: subscriber.sessionId,
    imageId: subscriber.imageId,
    sourceKey: subscriber.sourceKey,
    localSignature: subscriber.descriptor.localSignature || "",
    state: error?.code === "api_offline" || error?.code === "request_timeout" ? IMAGE_SCAN_STATE.apiOffline : IMAGE_SCAN_STATE.failed,
    error: {
      code: safeDisplayText(error?.code || "scan_failed"),
      message: safeDisplayText(error?.message || "Image scan failed."),
      retryable: Boolean(error?.retryable)
    }
  });
}

function addSubscriber(sourceKey, subscriber) {
  if (!subscribersBySourceKey.has(sourceKey)) {
    subscribersBySourceKey.set(sourceKey, new Map());
  }
  subscribersBySourceKey.get(sourceKey).set(subscriberKey(subscriber), subscriber);
}

function removeSubscriber(sourceKey, subscriber) {
  const subscribers = subscribersBySourceKey.get(sourceKey);
  if (!subscribers) {
    return;
  }
  subscribers.delete(subscriberKey(subscriber));
  if (subscribers.size === 0) {
    subscribersBySourceKey.delete(sourceKey);
  }
}

function cleanupSource(sourceKey) {
  subscribersBySourceKey.delete(sourceKey);
  pendingSourceKeys.delete(sourceKey);
}

function subscriberKey(subscriber) {
  return `${subscriber.tabId}:${subscriber.imageId}`;
}

async function loadBackendMetadata(options) {
  if (options.mockModeEnabled) {
    return backendMetadataFromHealth(mockHealthPayload());
  }
  const health = await checkHealth(options.apiBaseUrl, Math.min(options.requestTimeoutMs, 8000));
  return backendMetadataFromHealth(health);
}

async function hashBlob(blob) {
  try {
    const buffer = await blob.arrayBuffer();
    const digest = await crypto.subtle.digest("SHA-256", buffer);
    return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
  } catch {
    return "";
  }
}

function shouldRetryScanError(error) {
  if (error instanceof ExtensionRequestError) {
    return error.retryable && ["api_offline", "api_request_failed", "request_timeout", "concurrency_limit_exceeded"].includes(error.code);
  }
  return false;
}

async function scanStateForTab(tab) {
  const options = await getOptions();
  const mode = tab?.url ? await getSiteModeForUrl(tab.url) : SITE_MODE.manualOnly;
  const session = tab?.id ? sessionsByTab.get(tab.id) : null;
  return {
    ok: true,
    active: Boolean(session?.active),
    siteMode: mode,
    globalScanningStopped: options.globalScanningStopped,
    mockModeEnabled: options.mockModeEnabled,
    apiBaseUrl: options.apiBaseUrl,
    queue: scanQueue.stats()
  };
}

async function resolveCommandTab(message) {
  if (typeof message.tabId === "number") {
    return { id: message.tabId, url: message.pageUrl || message.url || "" };
  }
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  const tab = tabs[0];
  return tab ? { id: tab.id, url: tab.url || "" } : null;
}

async function ensureContentScript(tabId) {
  await chrome.scripting.executeScript({
    target: { tabId },
    files: ["content-script.js"]
  });
}

async function sendToTab(tabId, message) {
  try {
    await chrome.tabs.sendMessage(tabId, message);
  } catch {
    // Some pages reject content scripts or navigated before the message arrived.
  }
}

function publicScannerOptions(options) {
  return {
    minDisplayedWidth: options.minDisplayedWidth,
    minDisplayedHeight: options.minDisplayedHeight,
    minNaturalWidth: options.minNaturalWidth,
    minNaturalHeight: options.minNaturalHeight,
    scanRootMarginPx: options.scanRootMarginPx,
    overlaysEnabled: options.overlaysEnabled,
    diagnosticsEnabled: options.diagnosticsEnabled,
    showModelScores: options.showModelScores,
    mockModeEnabled: options.mockModeEnabled,
    apiBaseUrl: options.apiBaseUrl
  };
}

async function updateRecord(baseRecord, options, stage, message) {
  await saveAnalysisRecord(
    {
      ...baseRecord,
      status: REQUEST_STATUS.running,
      stage,
      stageMessage: message,
      timestamp: new Date().toISOString()
    },
    options.historyLimit
  );
}

async function notifyPage(tabId, frameId, message, state) {
  if (typeof tabId !== "number") {
    return;
  }
  const target = { tabId };
  if (typeof frameId === "number" && frameId >= 0) {
    target.frameIds = [frameId];
  }
  try {
    await chrome.scripting.executeScript({
      target,
      files: ["content-script.js"]
    });
    await chrome.tabs.sendMessage(tabId, {
      type: "DFD_STATUS",
      message,
      state
    });
  } catch {
    // Some pages do not allow script injection. The popup/results page still has persistent state.
  }
}

function errorToRecord(error) {
  if (error instanceof ExtensionRequestError) {
    return {
      code: error.code,
      message: error.message,
      status: error.status,
      retryable: error.retryable
    };
  }
  return {
    code: "extension_unhandled_error",
    message: safeDisplayText(error?.message || "Analysis failed."),
    status: null,
    retryable: false
  };
}

function notificationForError(error) {
  if (error.code === "api_offline" || error.code === "api_not_ready" || error.code === "request_timeout") {
    return "Local detector unavailable";
  }
  return "Analysis failed";
}

function sanitizeErrorForLog(error) {
  return {
    name: safeDisplayText(error?.name || "Error"),
    code: safeDisplayText(error?.code || ""),
    message: safeDisplayText(error?.message || "")
  };
}
