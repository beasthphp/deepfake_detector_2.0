import {
  API_FIELD_NAME,
  API_HEALTH_PATH,
  API_PREDICT_PATH,
  MAX_UPLOAD_BYTES,
  OFFLINE_MESSAGE
} from "./constants.js";
import {
  filenameFromUrl,
  isImageLikeContentType,
  mapApiError,
  validateHealthResponse,
  validateImageBlob,
  validatePredictionResponse,
  validateSelectedImageUrl
} from "./validation.js";

export class ExtensionRequestError extends Error {
  constructor(code, message, options = {}) {
    super(message);
    this.name = "ExtensionRequestError";
    this.code = code;
    this.status = options.status || null;
    this.retryable = Boolean(options.retryable);
  }
}

export async function checkHealth(apiBaseUrl, timeoutMs, fetchImpl = fetch) {
  let response;
  try {
    response = await fetchWithTimeout(`${apiBaseUrl}${API_HEALTH_PATH}`, { method: "GET" }, timeoutMs, fetchImpl);
  } catch (error) {
    throw normalizeFetchError(error, "api_offline", OFFLINE_MESSAGE, true);
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new ExtensionRequestError("malformed_health_json", "The local detector returned malformed health data.", {
      status: response.status,
      retryable: true
    });
  }

  if (!response.ok) {
    const mapped = mapApiError(response.status, payload);
    throw new ExtensionRequestError(mapped.code, mapped.message, { status: response.status, retryable: true });
  }

  const validation = validateHealthResponse(payload);
  if (!validation.ok) {
    throw new ExtensionRequestError(validation.code, validation.message, { retryable: true });
  }
  return payload;
}

export async function fetchSelectedImage(imageUrl, timeoutMs, fetchImpl = fetch) {
  const urlCheck = validateSelectedImageUrl(imageUrl);
  if (!urlCheck.ok) {
    throw new ExtensionRequestError(urlCheck.code, urlCheck.message);
  }

  if (urlCheck.scheme === "data:") {
    return dataUrlToBlobResult(imageUrl);
  }

  let response;
  try {
    response = await fetchWithTimeout(
      imageUrl,
      { method: "GET", cache: "no-store", credentials: "omit" },
      timeoutMs,
      fetchImpl
    );
  } catch (error) {
    throw normalizeFetchError(error, "image_download_failed", "Could not download the selected image.", true);
  }

  if (!response.ok) {
    throw new ExtensionRequestError(
      "image_download_http_error",
      response.status === 401 || response.status === 403
        ? "The selected image appears to be protected or authentication-gated."
        : "Could not download the selected image.",
      { status: response.status, retryable: true }
    );
  }

  const contentLength = Number(response.headers.get("content-length") || "0");
  if (Number.isFinite(contentLength) && contentLength > MAX_UPLOAD_BYTES) {
    throw new ExtensionRequestError("image_too_large", "Selected image is larger than the 10 MB API limit.", {
      status: 413
    });
  }

  const contentType = response.headers.get("content-type") || "";
  if (!isImageLikeContentType(contentType)) {
    throw new ExtensionRequestError("unsupported_image_content_type", "Selected resource is not a supported image.");
  }

  const blob = await response.blob();
  const validation = validateImageBlob(blob);
  if (!validation.ok) {
    throw new ExtensionRequestError(validation.code, validation.message);
  }

  return {
    blob,
    filename: filenameFromUrl(imageUrl, blob.type),
    contentType: blob.type
  };
}

export async function predictImage(apiBaseUrl, imageBlob, filename, timeoutMs, fetchImpl = fetch) {
  const validation = validateImageBlob(imageBlob);
  if (!validation.ok) {
    throw new ExtensionRequestError(validation.code, validation.message);
  }

  const formData = new FormData();
  formData.append(API_FIELD_NAME, imageBlob, filename || "selected-image.jpg");

  let response;
  try {
    response = await fetchWithTimeout(
      `${apiBaseUrl}${API_PREDICT_PATH}`,
      { method: "POST", body: formData },
      timeoutMs,
      fetchImpl
    );
  } catch (error) {
    throw normalizeFetchError(error, "api_request_failed", "The local detector request failed.", true);
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new ExtensionRequestError("malformed_prediction_json", "The local detector returned malformed JSON.", {
      status: response.status,
      retryable: true
    });
  }

  if (!response.ok) {
    const mapped = mapApiError(response.status, payload);
    throw new ExtensionRequestError(mapped.code, mapped.message, {
      status: response.status,
      retryable: response.status >= 429
    });
  }

  const responseCheck = validatePredictionResponse(payload);
  if (!responseCheck.ok) {
    throw new ExtensionRequestError(responseCheck.code, responseCheck.message, { retryable: true });
  }

  return payload;
}

export async function fetchWithTimeout(url, options, timeoutMs, fetchImpl = fetch) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetchImpl(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
}

async function dataUrlToBlobResult(imageUrl) {
  let response;
  try {
    response = await fetch(imageUrl);
  } catch {
    throw new ExtensionRequestError("data_url_decode_failed", "Could not decode the selected data image.");
  }
  const blob = await response.blob();
  const validation = validateImageBlob(blob);
  if (!validation.ok) {
    throw new ExtensionRequestError(validation.code, validation.message);
  }
  return {
    blob,
    filename: filenameFromUrl(imageUrl, blob.type),
    contentType: blob.type
  };
}

function normalizeFetchError(error, code, fallbackMessage, retryable) {
  if (error?.name === "AbortError") {
    return new ExtensionRequestError("request_timeout", "The request timed out.", { status: 504, retryable: true });
  }
  return new ExtensionRequestError(code, fallbackMessage, { retryable });
}
