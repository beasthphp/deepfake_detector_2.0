import { RESULT_NOTICE, SCAN_COMMAND, SITE_MODE, STORAGE_KEYS } from "../shared/constants.js";
import { checkHealth } from "../shared/api-client.js";
import { clearStoredResults, getCurrentRequest, getOptions } from "../shared/storage.js";
import { isSupportedPageUrl } from "../shared/scanner.js";
import { buildResultSummary, formatPercent, truncateMiddle } from "../shared/validation.js";

const elements = {
  apiOrigin: document.getElementById("apiOrigin"),
  apiStatus: document.getElementById("apiStatus"),
  requestStatus: document.getElementById("requestStatus"),
  selectedImage: document.getElementById("selectedImage"),
  faceSummary: document.getElementById("faceSummary"),
  faceList: document.getElementById("faceList"),
  notice: document.getElementById("notice"),
  scanStatus: document.getElementById("scanStatus"),
  siteMode: document.getElementById("siteMode"),
  scanVisibleImages: document.getElementById("scanVisibleImages"),
  stopScanning: document.getElementById("stopScanning"),
  rescanPage: document.getElementById("rescanPage"),
  clearPageResults: document.getElementById("clearPageResults"),
  enableSite: document.getElementById("enableSite"),
  disableSite: document.getElementById("disableSite"),
  clearPredictionCache: document.getElementById("clearPredictionCache"),
  globalStop: document.getElementById("globalStop"),
  resumeGlobal: document.getElementById("resumeGlobal"),
  openResults: document.getElementById("openResults"),
  clearResults: document.getElementById("clearResults"),
  openOptions: document.getElementById("openOptions")
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  elements.notice.textContent = RESULT_NOTICE;
  elements.openResults.addEventListener("click", () => {
    chrome.tabs.create({ url: chrome.runtime.getURL("results/results.html") });
  });
  elements.clearResults.addEventListener("click", async () => {
    await clearStoredResults();
    await render();
  });
  elements.openOptions.addEventListener("click", () => {
    chrome.runtime.openOptionsPage();
  });
  elements.scanVisibleImages.addEventListener("click", () => sendScanCommand(SCAN_COMMAND.start));
  elements.stopScanning.addEventListener("click", () => sendScanCommand(SCAN_COMMAND.stop));
  elements.rescanPage.addEventListener("click", () => sendScanCommand(SCAN_COMMAND.rescan));
  elements.clearPageResults.addEventListener("click", () => sendScanCommand(SCAN_COMMAND.clearPage));
  elements.enableSite.addEventListener("click", () => sendScanCommand(SCAN_COMMAND.enableSite));
  elements.disableSite.addEventListener("click", () => sendScanCommand(SCAN_COMMAND.disableSite));
  elements.clearPredictionCache.addEventListener("click", () => sendScanCommand(SCAN_COMMAND.clearCache));
  elements.globalStop.addEventListener("click", () => sendScanCommand(SCAN_COMMAND.globalStop));
  elements.resumeGlobal.addEventListener("click", () => sendScanCommand(SCAN_COMMAND.resumeGlobal));
  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName === "local" && changes[STORAGE_KEYS.currentRequest]) {
      render();
    }
    if (areaName === "local" && (changes[STORAGE_KEYS.options] || changes[STORAGE_KEYS.siteSettings])) {
      renderScanState();
    }
  });
  await render();
  await renderScanState();
  checkApiStatus();
}

async function checkApiStatus() {
  const options = await getOptions();
  elements.apiOrigin.textContent = options.apiBaseUrl;
  try {
    await checkHealth(options.apiBaseUrl, Math.min(options.requestTimeoutMs, 5000));
    elements.apiStatus.textContent = "Online";
    elements.apiStatus.classList.add("online");
    elements.apiStatus.classList.remove("offline");
  } catch {
    elements.apiStatus.textContent = "Offline";
    elements.apiStatus.classList.add("offline");
    elements.apiStatus.classList.remove("online");
  }
}

async function render() {
  const record = await getCurrentRequest();
  const options = await getOptions();
  elements.faceList.replaceChildren();
  if (!record) {
    elements.requestStatus.textContent = "No analysis yet";
    elements.selectedImage.textContent = "";
    elements.faceSummary.textContent = "";
    return;
  }

  elements.requestStatus.textContent = record.error?.message || record.stageMessage || record.status;
  elements.selectedImage.textContent = record.selectedImageUrl
    ? `Image: ${truncateMiddle(record.selectedImageUrl, 90)}`
    : "";

  if (record.response) {
    elements.faceSummary.textContent = buildResultSummary(record.response);
    renderFaces(record.response.faces || [], options.showModelScores);
  } else {
    elements.faceSummary.textContent = record.status === "failed" ? "Analysis failed" : "Analysis in progress";
  }
}

async function sendScanCommand(command) {
  const tab = await activeTab();
  elements.scanStatus.textContent = "Updating...";
  try {
    const response = await chrome.runtime.sendMessage({
      type: "DFD_SCAN_COMMAND",
      command,
      tabId: tab?.id,
      pageUrl: tab?.url || ""
    });
    if (!response?.ok) {
      elements.scanStatus.textContent = response?.error?.message || "Scan command failed";
    } else {
      elements.scanStatus.textContent = response.message || "Scan command completed";
    }
  } catch (error) {
    elements.scanStatus.textContent = error?.message || "Scan command failed";
  }
  await renderScanState();
}

async function renderScanState() {
  const tab = await activeTab();
  const supported = Boolean(tab?.url && isSupportedPageUrl(tab.url));
  let scanState = {
    active: false,
    siteMode: SITE_MODE.manualOnly,
    globalScanningStopped: false,
    mockModeEnabled: false,
    apiBaseUrl: ""
  };

  if (supported) {
    try {
      const response = await chrome.runtime.sendMessage({
        type: "DFD_SCAN_COMMAND",
        command: SCAN_COMMAND.getState,
        tabId: tab.id,
        pageUrl: tab.url
      });
      if (response?.ok) {
        scanState = { ...scanState, ...response };
      }
    } catch {
      // The status line below remains conservative if the service worker is restarting.
    }
  }

  if (!supported) {
    elements.scanStatus.textContent = "This page cannot be scanned";
    elements.siteMode.textContent = "";
  } else if (scanState.globalScanningStopped) {
    elements.scanStatus.textContent = "Global scanning stopped";
    elements.siteMode.textContent = `Site: ${siteModeLabel(scanState.siteMode)}`;
  } else {
    const mock = scanState.mockModeEnabled ? " · Mock mode" : "";
    elements.scanStatus.textContent = scanState.active ? `Scanning visible images${mock}` : `Not scanning${mock}`;
    elements.siteMode.textContent = `Site: ${siteModeLabel(scanState.siteMode)} · API: ${scanState.apiBaseUrl}`;
  }

  const canScan = supported && !scanState.globalScanningStopped && scanState.siteMode !== SITE_MODE.disabled;
  elements.scanVisibleImages.disabled = !canScan;
  elements.rescanPage.disabled = !canScan;
  elements.stopScanning.disabled = !supported || !scanState.active;
  elements.clearPageResults.disabled = !supported;
  elements.enableSite.disabled = !supported || scanState.siteMode === SITE_MODE.enabled || scanState.globalScanningStopped;
  elements.disableSite.disabled = !supported || scanState.siteMode === SITE_MODE.disabled;
  elements.clearPredictionCache.disabled = false;
  elements.globalStop.disabled = scanState.globalScanningStopped;
  elements.resumeGlobal.disabled = !scanState.globalScanningStopped;
}

function renderFaces(faces, showScores) {
  const limited = faces.slice(0, 3);
  for (const face of limited) {
    const row = document.createElement("div");
    row.className = "face-row";

    const title = document.createElement("strong");
    title.textContent = `Face ${Number(face.face_index) + 1}: ${face.label}`;
    row.append(title);

    if (showScores) {
      const scores = document.createElement("span");
      scores.textContent = `Real ${formatPercent(face.real_score)} / Fake ${formatPercent(face.fake_score)}`;
      row.append(scores);
    }

    elements.faceList.append(row);
  }
  if (faces.length > limited.length) {
    const more = document.createElement("p");
    more.className = "muted";
    more.textContent = `${faces.length - limited.length} more face result(s) in the full results page.`;
    elements.faceList.append(more);
  }
}

async function activeTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0] || null;
}

function siteModeLabel(mode) {
  if (mode === SITE_MODE.enabled) {
    return "enabled";
  }
  if (mode === SITE_MODE.disabled) {
    return "disabled";
  }
  return "manual only";
}
