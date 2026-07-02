export const MENU_ID = "analyze-face-for-deepfake";

export const API_FIELD_NAME = "file";
export const API_HEALTH_PATH = "/health";
export const API_PREDICT_PATH = "/predict";

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
export const SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
export const SUPPORTED_IMAGE_EXTENSIONS = {
  "image/jpeg": ".jpg",
  "image/png": ".png",
  "image/webp": ".webp"
};

export const STORAGE_KEYS = {
  options: "dfd_options",
  currentRequest: "dfd_current_request",
  history: "dfd_history",
  predictionCache: "dfd_prediction_cache",
  siteSettings: "dfd_site_settings"
};

export const DEFAULT_OPTIONS = {
  apiBaseUrl: "http://127.0.0.1:8000",
  requestTimeoutMs: 30000,
  autoOpenResults: true,
  showModelScores: true,
  historyLimit: 10,
  minDisplayedWidth: 128,
  minDisplayedHeight: 128,
  minNaturalWidth: 128,
  minNaturalHeight: 128,
  scanRootMarginPx: 300,
  scanRequestConcurrency: 2,
  scanQueueLimit: 50,
  predictionCacheLimit: 200,
  overlaysEnabled: true,
  diagnosticsEnabled: false,
  mockModeEnabled: false,
  mockScenario: "likely_fake",
  globalScanningStopped: false
};

export const SITE_MODE = {
  enabled: "enabled",
  disabled: "disabled",
  manualOnly: "manual_only"
};

export const DEFAULT_BACKEND_METADATA = {
  model: {
    id: "existing-cnn-93acc",
    version: "1"
  },
  detector: {
    id: "opencv_haar_frontalface_default",
    version: "1"
  },
  cropStrategy: {
    id: "preserve_or_context_full_head_l40_r40_t70_b35_square",
    version: "phase2.1"
  },
  thresholds: {
    id: "score-thresholds",
    version: "1"
  }
};

export const SCAN_COMMAND = {
  start: "start_scan",
  stop: "stop_scan",
  rescan: "rescan_page",
  clearPage: "clear_page_results",
  clearCache: "clear_prediction_cache",
  enableSite: "enable_site",
  disableSite: "disable_site",
  globalStop: "global_stop",
  resumeGlobal: "resume_global",
  getState: "get_scan_state"
};

export const IMAGE_SCAN_STATE = {
  notScanned: "not_scanned",
  queued: "queued",
  downloading: "downloading",
  analysing: "analysing",
  completed: "completed",
  noFaceDetected: "no_face_detected",
  unsupported: "unsupported",
  failed: "failed",
  apiOffline: "api_offline"
};

export const CACHE_STATUS = {
  completed: "completed",
  noFaceDetected: "no_face_detected",
  unsupported: "unsupported",
  temporaryFailure: "temporary_failure"
};

export const MOCK_SCENARIOS = {
  likelyFake: "likely_fake",
  likelyReal: "likely_real",
  uncertain: "uncertain",
  multipleFaces: "multiple_faces",
  noFace: "no_face",
  timeout: "timeout",
  malformed: "malformed",
  apiError: "api_error"
};

export const REQUEST_STATUS = {
  idle: "idle",
  running: "running",
  completed: "completed",
  noFaceDetected: "no_face_detected",
  failed: "failed"
};

export const REQUEST_STAGE = {
  queued: "queued",
  checkingApi: "checking_api",
  downloadingImage: "downloading_selected_image",
  detectingFaces: "detecting_faces",
  analyzingFaces: "analyzing_detected_faces",
  completed: "completed",
  failed: "failed"
};

export const STAGE_MESSAGES = {
  queued: "Preparing selected image analysis...",
  checking_api: "Checking local API...",
  downloading_selected_image: "Downloading selected image...",
  detecting_faces: "Detecting faces...",
  analyzing_detected_faces: "Analysing detected faces...",
  completed: "Analysis completed",
  failed: "Analysis failed"
};

export const OFFLINE_MESSAGE = [
  "Local detector is not running.",
  "",
  "Start it with:",
  "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
].join("\n");

export const RESULT_NOTICE =
  "This is a probabilistic model result, not proof that an image is authentic or manipulated.";

export const EXPERIMENTAL_NOTICE =
  "Experimental detector. Current real-world predictions are unreliable and probabilistic.";
