import { RESULT_NOTICE, STORAGE_KEYS } from "../shared/constants.js";
import { clearStoredResults, getCurrentRequest, getOptions } from "../shared/storage.js";
import {
  buildResultSummary,
  formatPercent,
  formatScore,
  isSafeOriginalImageLink,
  originFromUrl,
  safeDisplayText,
  truncateMiddle
} from "../shared/validation.js";

const elements = {
  statusLine: document.getElementById("statusLine"),
  summaryText: document.getElementById("summaryText"),
  notice: document.getElementById("notice"),
  requestDetails: document.getElementById("requestDetails"),
  timingDetails: document.getElementById("timingDetails"),
  faces: document.getElementById("faces"),
  warnings: document.getElementById("warnings"),
  clearHistory: document.getElementById("clearHistory"),
  originalImageLink: document.getElementById("originalImageLink")
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  elements.notice.textContent = RESULT_NOTICE;
  elements.clearHistory.addEventListener("click", async () => {
    await clearStoredResults();
    await render();
  });
  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName === "local" && changes[STORAGE_KEYS.currentRequest]) {
      render();
    }
  });
  await render();
}

async function render() {
  const record = await getCurrentRequest();
  const options = await getOptions();
  elements.requestDetails.replaceChildren();
  elements.timingDetails.replaceChildren();
  elements.faces.replaceChildren();
  elements.warnings.replaceChildren();

  if (!record) {
    elements.statusLine.textContent = "No request stored";
    elements.summaryText.textContent = "No analysis yet";
    elements.originalImageLink.classList.add("hidden");
    return;
  }

  elements.statusLine.textContent = record.stageMessage || record.status;
  elements.summaryText.textContent = record.error?.message || buildResultSummary(record.response);
  renderRequestDetails(record);
  renderTiming(record.response?.timing_ms);
  renderFaces(record.response, options.showModelScores);
  renderWarnings(record.response?.warnings || []);
  renderOriginalLink(record.selectedImageUrl);
}

function renderRequestDetails(record) {
  const response = record.response;
  const image = response?.image || {};
  appendDetail(elements.requestDetails, "Status", record.status);
  appendDetail(elements.requestDetails, "Request ID", response?.request_id || record.requestId || "n/a");
  appendDetail(elements.requestDetails, "API origin", record.apiBaseUrl || "n/a");
  appendDetail(elements.requestDetails, "Page origin", originFromUrl(record.pageUrl));
  appendDetail(elements.requestDetails, "Image origin", originFromUrl(record.selectedImageUrl));
  appendDetail(elements.requestDetails, "Selected image", truncateMiddle(record.selectedImageUrl || "n/a", 120));
  appendDetail(elements.requestDetails, "Image dimensions", image.width && image.height ? `${image.width} x ${image.height}` : "n/a");
  appendDetail(elements.requestDetails, "Image format", image.format || "n/a");
  appendDetail(elements.requestDetails, "Faces detected", String(response?.faces_detected ?? "n/a"));
  appendDetail(elements.requestDetails, "Model", versionLabel(response?.model));
  appendDetail(elements.requestDetails, "Detector", versionLabel(response?.detector));
  appendDetail(elements.requestDetails, "Crop strategy", versionLabel(response?.crop_strategy));
  appendDetail(elements.requestDetails, "Thresholds", versionLabel(response?.thresholds));
  appendDetail(elements.requestDetails, "Timestamp", record.timestamp || "n/a");
}

function renderTiming(timing = {}) {
  appendDetail(elements.timingDetails, "Decode", `${formatMs(timing.decode)} ms`);
  appendDetail(elements.timingDetails, "Detection", `${formatMs(timing.face_detection)} ms`);
  appendDetail(elements.timingDetails, "Crop/prep", `${formatMs(timing.crop_preprocessing)} ms`);
  appendDetail(elements.timingDetails, "Classification", `${formatMs(timing.classification)} ms`);
  appendDetail(elements.timingDetails, "Serialization", `${formatMs(timing.serialization)} ms`);
  appendDetail(elements.timingDetails, "Total", `${formatMs(timing.total)} ms`);
}

function renderFaces(response, showScores) {
  if (!response) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "No face results are available.";
    elements.faces.append(p);
    return;
  }

  if (response.status === "no_face_detected") {
    const p = document.createElement("p");
    p.textContent = "No supported human face was detected. The deepfake classifier was not run. This does not mean the image is real.";
    elements.faces.append(p);
    return;
  }

  for (const face of response.faces || []) {
    elements.faces.append(createFaceCard(face, showScores));
  }
}

function createFaceCard(face, showScores) {
  const card = document.createElement("article");
  card.className = "face-card";

  const title = document.createElement("div");
  title.className = "face-title";

  const heading = document.createElement("h3");
  heading.textContent = `Face ${Number(face.face_index) + 1}`;
  title.append(heading);

  const label = document.createElement("span");
  label.className = `label ${labelClass(face.label)}`;
  label.textContent = face.label;
  title.append(label);

  card.append(title);

  const details = document.createElement("dl");
  if (showScores) {
    appendDetail(details, "Real score", `${formatPercent(face.real_score)} (${formatScore(face.real_score)})`);
    appendDetail(details, "Fake score", `${formatPercent(face.fake_score)} (${formatScore(face.fake_score)})`);
  } else {
    appendDetail(details, "Model scores", "hidden by extension option");
  }
  appendDetail(details, "Detection score", face.face_detection_score == null ? "n/a" : `${formatPercent(face.face_detection_score)} (${formatScore(face.face_detection_score)})`);
  appendDetail(details, "Bounding box", formatBox(face.bounding_box));
  appendDetail(details, "Crop box", formatBox(face.crop_box));
  appendDetail(details, "Crop strategy", face.crop_strategy || "n/a");
  appendDetail(details, "Original preserved", face.preserved_original ? "yes" : "no");
  card.append(details);
  return card;
}

function renderWarnings(warnings) {
  if (!warnings.length) {
    const item = document.createElement("li");
    item.textContent = "No API warnings returned.";
    elements.warnings.append(item);
    return;
  }
  for (const warning of warnings) {
    const item = document.createElement("li");
    item.textContent = safeDisplayText(warning);
    elements.warnings.append(item);
  }
}

function renderOriginalLink(url) {
  if (isSafeOriginalImageLink(url)) {
    elements.originalImageLink.href = url;
    elements.originalImageLink.textContent = `Open original image at ${formatOrigin(url)}`;
    elements.originalImageLink.classList.remove("hidden");
  } else {
    elements.originalImageLink.removeAttribute("href");
    elements.originalImageLink.classList.add("hidden");
  }
}

function appendDetail(list, key, value) {
  const dt = document.createElement("dt");
  dt.textContent = key;
  const dd = document.createElement("dd");
  dd.textContent = safeDisplayText(value);
  list.append(dt, dd);
}

function formatMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "0.000";
}

function formatBox(box) {
  if (!box) {
    return "n/a";
  }
  return `x1 ${box.x1}, y1 ${box.y1}, x2 ${box.x2}, y2 ${box.y2}`;
}

function labelClass(label) {
  if (label === "Likely Fake") {
    return "fake";
  }
  if (label === "Likely Real") {
    return "real";
  }
  return "uncertain";
}

function formatOrigin(url) {
  return originFromUrl(url);
}

function versionLabel(component) {
  if (!component) {
    return "n/a";
  }
  return `${component.id || "unknown"} @ ${component.version || "unknown"}`;
}
