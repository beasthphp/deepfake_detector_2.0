import { DEFAULT_OPTIONS } from "../shared/constants.js";
import { clearPredictionCache } from "../shared/cache.js";
import { getOptions, saveOptions } from "../shared/storage.js";
import { normalizeOptions, validateApiBaseUrl } from "../shared/validation.js";

const form = document.getElementById("optionsForm");
const fields = {
  apiBaseUrl: document.getElementById("apiBaseUrl"),
  requestTimeoutSeconds: document.getElementById("requestTimeoutSeconds"),
  autoOpenResults: document.getElementById("autoOpenResults"),
  showModelScores: document.getElementById("showModelScores"),
  historyLimit: document.getElementById("historyLimit"),
  minDisplayedWidth: document.getElementById("minDisplayedWidth"),
  minDisplayedHeight: document.getElementById("minDisplayedHeight"),
  minNaturalWidth: document.getElementById("minNaturalWidth"),
  minNaturalHeight: document.getElementById("minNaturalHeight"),
  scanRootMarginPx: document.getElementById("scanRootMarginPx"),
  scanRequestConcurrency: document.getElementById("scanRequestConcurrency"),
  scanQueueLimit: document.getElementById("scanQueueLimit"),
  predictionCacheLimit: document.getElementById("predictionCacheLimit"),
  overlaysEnabled: document.getElementById("overlaysEnabled"),
  diagnosticsEnabled: document.getElementById("diagnosticsEnabled"),
  mockModeEnabled: document.getElementById("mockModeEnabled"),
  mockScenario: document.getElementById("mockScenario"),
  globalScanningStopped: document.getElementById("globalScanningStopped"),
  message: document.getElementById("message"),
  resetDefaults: document.getElementById("resetDefaults"),
  clearPredictionCache: document.getElementById("clearPredictionCache")
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  populate(await getOptions());
  form.addEventListener("submit", save);
  fields.resetDefaults.addEventListener("click", async () => {
    const defaults = normalizeOptions(DEFAULT_OPTIONS);
    await saveOptions(defaults);
    populate(defaults);
    showMessage("Defaults restored.");
  });
  fields.clearPredictionCache.addEventListener("click", async () => {
    await clearPredictionCache();
    showMessage("Prediction cache cleared.");
  });
}

async function save(event) {
  event.preventDefault();
  const apiValidation = validateApiBaseUrl(fields.apiBaseUrl.value);
  if (!apiValidation.ok) {
    showMessage(apiValidation.message, true);
    return;
  }

  const options = normalizeOptions({
    apiBaseUrl: fields.apiBaseUrl.value,
    requestTimeoutMs: Number(fields.requestTimeoutSeconds.value) * 1000,
    autoOpenResults: fields.autoOpenResults.checked,
    showModelScores: fields.showModelScores.checked,
    historyLimit: Number(fields.historyLimit.value),
    minDisplayedWidth: Number(fields.minDisplayedWidth.value),
    minDisplayedHeight: Number(fields.minDisplayedHeight.value),
    minNaturalWidth: Number(fields.minNaturalWidth.value),
    minNaturalHeight: Number(fields.minNaturalHeight.value),
    scanRootMarginPx: Number(fields.scanRootMarginPx.value),
    scanRequestConcurrency: Number(fields.scanRequestConcurrency.value),
    scanQueueLimit: Number(fields.scanQueueLimit.value),
    predictionCacheLimit: Number(fields.predictionCacheLimit.value),
    overlaysEnabled: fields.overlaysEnabled.checked,
    diagnosticsEnabled: fields.diagnosticsEnabled.checked,
    mockModeEnabled: fields.mockModeEnabled.checked,
    mockScenario: fields.mockScenario.value,
    globalScanningStopped: fields.globalScanningStopped.checked
  });
  await saveOptions(options);
  populate(options);
  const warning = apiValidation.warnings?.length ? ` ${apiValidation.warnings.join(" ")}` : "";
  showMessage(`Options saved.${warning}`);
}

function populate(options) {
  fields.apiBaseUrl.value = options.apiBaseUrl;
  fields.requestTimeoutSeconds.value = String(Math.round(options.requestTimeoutMs / 1000));
  fields.autoOpenResults.checked = options.autoOpenResults;
  fields.showModelScores.checked = options.showModelScores;
  fields.historyLimit.value = String(options.historyLimit);
  fields.minDisplayedWidth.value = String(options.minDisplayedWidth);
  fields.minDisplayedHeight.value = String(options.minDisplayedHeight);
  fields.minNaturalWidth.value = String(options.minNaturalWidth);
  fields.minNaturalHeight.value = String(options.minNaturalHeight);
  fields.scanRootMarginPx.value = String(options.scanRootMarginPx);
  fields.scanRequestConcurrency.value = String(options.scanRequestConcurrency);
  fields.scanQueueLimit.value = String(options.scanQueueLimit);
  fields.predictionCacheLimit.value = String(options.predictionCacheLimit);
  fields.overlaysEnabled.checked = options.overlaysEnabled;
  fields.diagnosticsEnabled.checked = options.diagnosticsEnabled;
  fields.mockModeEnabled.checked = options.mockModeEnabled;
  fields.mockScenario.value = options.mockScenario;
  fields.globalScanningStopped.checked = options.globalScanningStopped;
}

function showMessage(message, isError = false) {
  fields.message.textContent = message;
  fields.message.classList.toggle("error", isError);
}
