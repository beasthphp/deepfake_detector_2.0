import { MOCK_SCENARIOS } from "./constants.js";
import { ExtensionRequestError } from "./api-client.js";

const MOCK_METADATA = {
  model: { id: "mock-extension-api", version: "phase5-test" },
  detector: { id: "mock-face-detector", version: "phase5-test" },
  crop_strategy: { id: "mock-crop-strategy", version: "phase5-test" },
  thresholds: { id: "mock-thresholds", version: "phase5-test", likely_fake_below: 0.4, likely_real_above: 0.6 }
};

export async function predictMockImage({ scenario = MOCK_SCENARIOS.likelyFake, imageBlob, descriptor = {}, timeoutMs = 30000 } = {}) {
  if (scenario === MOCK_SCENARIOS.timeout) {
    await delay(Math.min(25, Math.max(1, Number(timeoutMs) || 1)));
    throw new ExtensionRequestError("request_timeout", "Mock API timeout.", { status: 504, retryable: true });
  }
  if (scenario === MOCK_SCENARIOS.apiError) {
    throw new ExtensionRequestError("mock_api_error", "Mock API error.", { status: 500, retryable: false });
  }
  if (scenario === MOCK_SCENARIOS.malformed) {
    return { status: "completed", faces: "not an array", mock: true };
  }

  const image = {
    width: positiveInteger(descriptor.naturalWidth) || 800,
    height: positiveInteger(descriptor.naturalHeight) || 600,
    format: formatFromBlob(imageBlob)
  };

  if (scenario === MOCK_SCENARIOS.noFace) {
    return baseResponse({
      status: "no_face_detected",
      image,
      faces: [],
      warnings: ["Mock mode: no face detected."]
    });
  }

  const faces =
    scenario === MOCK_SCENARIOS.multipleFaces
      ? [
          face(0, image, 0.15, "Likely Fake", [0.1, 0.15, 0.32, 0.48]),
          face(1, image, 0.52, "Uncertain", [0.48, 0.12, 0.7, 0.42]),
          face(2, image, 0.84, "Likely Real", [0.28, 0.52, 0.47, 0.82])
        ]
      : [singleFaceForScenario(scenario, image)];

  return baseResponse({
    status: "completed",
    image,
    faces,
    warnings: ["Mock mode: deterministic extension test response."]
  });
}

export function mockHealthPayload() {
  return {
    status: "ok",
    model_loaded: true,
    detector_loaded: true,
    device: "mock",
    model_id: MOCK_METADATA.model.id,
    model_version: MOCK_METADATA.model.version,
    detector_id: MOCK_METADATA.detector.id,
    detector_version: MOCK_METADATA.detector.version,
    crop_strategy_id: MOCK_METADATA.crop_strategy.id,
    crop_strategy_version: MOCK_METADATA.crop_strategy.version,
    threshold_id: MOCK_METADATA.thresholds.id,
    threshold_version: MOCK_METADATA.thresholds.version
  };
}

function singleFaceForScenario(scenario, image) {
  if (scenario === MOCK_SCENARIOS.likelyReal) {
    return face(0, image, 0.86, "Likely Real", [0.32, 0.18, 0.56, 0.56]);
  }
  if (scenario === MOCK_SCENARIOS.uncertain) {
    return face(0, image, 0.51, "Uncertain", [0.32, 0.18, 0.56, 0.56]);
  }
  return face(0, image, 0.12, "Likely Fake", [0.32, 0.18, 0.56, 0.56]);
}

function baseResponse({ status, image, faces, warnings }) {
  return {
    request_id: `mock-${Date.now()}`,
    status,
    ...MOCK_METADATA,
    image,
    faces_detected: faces.length,
    faces,
    timing_ms: {
      decode: 0,
      face_detection: 1,
      crop_preprocessing: 1,
      classification: 1,
      serialization: 0,
      total: 3
    },
    warnings,
    mock: true
  };
}

function face(index, image, realScore, label, normalizedBox) {
  const [x1, y1, x2, y2] = normalizedBox;
  const box = {
    x1: Math.round(image.width * x1),
    y1: Math.round(image.height * y1),
    x2: Math.round(image.width * x2),
    y2: Math.round(image.height * y2)
  };
  return {
    face_index: index,
    bounding_box: box,
    crop_box: box,
    face_detection_score: 0.95,
    crop_strategy: MOCK_METADATA.crop_strategy.id,
    preserved_original: true,
    real_score: realScore,
    fake_score: Number((1 - realScore).toFixed(4)),
    label
  };
}

function formatFromBlob(blob) {
  const type = String(blob?.type || "image/jpeg").toLowerCase();
  if (type.includes("png")) {
    return "PNG";
  }
  if (type.includes("webp")) {
    return "WEBP";
  }
  return "JPEG";
}

function positiveInteger(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.round(number) : 0;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
