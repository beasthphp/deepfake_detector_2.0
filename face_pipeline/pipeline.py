from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from face_pipeline.classifier import DEFAULT_MODEL_PATH, DeepfakeClassifier
from face_pipeline.cropper import CropConfig, crop_face
from face_pipeline.detector import FaceDetector, FaceDetectorConfig, ImageInput, load_rgb_image
from face_pipeline.visualization import draw_annotations


class DeepfakeFacePipeline:
    """Detect faces, crop them, and classify each crop with the existing model."""

    def __init__(
        self,
        detector: FaceDetector | None = None,
        classifier: DeepfakeClassifier | None = None,
        crop_config: CropConfig | None = None,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.classifier = classifier or DeepfakeClassifier()
        self.crop_config = crop_config or CropConfig()

    def analyze(
        self,
        image: ImageInput,
        image_name: str | None = None,
        output_dir: str | Path | None = None,
        save_crops: bool = False,
        save_annotated: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        pil_image = load_rgb_image(image)
        display_name = image_name or _image_display_name(image)

        detector_started = time.perf_counter()
        detections = self.detector.detect(pil_image)
        detector_ms = _elapsed_ms(detector_started)

        if not detections:
            result: dict[str, Any] = {
                "image": display_name,
                "faces_detected": 0,
                "status": "no_face_detected",
                "detector": "opencv_haar_frontalface_default",
                "detector_time_ms": detector_ms,
            }
            result["processing_time_ms"] = _elapsed_ms(started)
            if output_dir is not None:
                _write_results(output_dir, result)
            return result

        output_path = Path(output_dir) if output_dir is not None else None
        faces = []
        classification_times = []
        for detection in detections:
            face_index = int(detection["face_index"])
            face_record: dict[str, Any] = {
                "face_index": face_index,
                "bounding_box": _public_box(detection),
                "detection_confidence": float(detection["confidence"]),
                "image_width": int(detection["image_width"]),
                "image_height": int(detection["image_height"]),
            }
            try:
                crop, crop_box = crop_face(pil_image, detection, self.crop_config)
                face_record["crop_box"] = crop_box
                face_record["crop_width"] = crop.width
                face_record["crop_height"] = crop.height

                if save_crops and output_path is not None:
                    output_path.mkdir(parents=True, exist_ok=True)
                    crop.save(output_path / f"face_{face_index}.jpg")

                classify_started = time.perf_counter()
                prediction = self.classifier.predict_image(crop)
                classification_ms = _elapsed_ms(classify_started)
                classification_times.append(classification_ms)
                face_record.update(
                    {
                        "real_score": round(float(prediction["real_score"]), 6),
                        "fake_score": round(float(prediction["fake_score"]), 6),
                        "label": prediction["label"],
                        "classification_time_ms": classification_ms,
                    }
                )
            except Exception as exc:
                face_record.update({"status": "face_processing_failed", "error": repr(exc)})
            faces.append(face_record)

        result = {
            "image": display_name,
            "faces_detected": len(detections),
            "status": "ok",
            "detector": "opencv_haar_frontalface_default",
            "preprocessing": {
                "color_mode": "RGB",
                "resize": [256, 256],
                "interpolation": "nearest",
                "dtype": "float32",
                "normalization": "divide_by_255",
                "channels": "last",
            },
            "thresholds": {
                "likely_fake": "real_score < 0.40",
                "uncertain": "0.40 <= real_score <= 0.60",
                "likely_real": "real_score > 0.60",
                "calibration_note": "provisional validation-derived ranges, not calibrated real-world confidence",
            },
            "detector_time_ms": detector_ms,
            "average_classification_time_ms": round(sum(classification_times) / len(classification_times), 3)
            if classification_times
            else None,
            "faces": faces,
        }
        result["processing_time_ms"] = _elapsed_ms(started)

        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)
            _write_results(output_path, result)
            if save_annotated:
                draw_annotations(pil_image, result, output_path / "annotated.jpg")

        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 face-crop deepfake classification.")
    parser.add_argument("--image", required=True, type=Path, help="Image file to analyze.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output directory for JSON/annotated/crops.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Explicit model artifact path.")
    parser.add_argument("--face-confidence", type=float, default=0.50, help="Detector confidence threshold in [0, 1].")
    parser.add_argument("--minimum-face-size", type=int, default=32, help="Minimum detector face size in pixels.")
    parser.add_argument("--crop-margin", type=float, default=0.20, help="Crop margin around each face.")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped face images under --output.")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated.jpg under --output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = DeepfakeFacePipeline(
        detector=FaceDetector(
            FaceDetectorConfig(
                confidence_threshold=args.face_confidence,
                minimum_face_size=args.minimum_face_size,
            )
        ),
        classifier=DeepfakeClassifier(model_path=args.model),
        crop_config=CropConfig(margin=args.crop_margin),
    )
    result = pipeline.analyze(
        args.image,
        output_dir=args.output,
        save_crops=args.save_crops,
        save_annotated=args.save_annotated,
    )
    print(json.dumps(result, indent=2))
    return 0


def _public_box(detection: dict[str, Any]) -> dict[str, int]:
    return {key: int(detection[key]) for key in ("x1", "y1", "x2", "y2")}


def _image_display_name(image: ImageInput) -> str:
    if isinstance(image, (str, Path)):
        return Path(image).name
    return "<in-memory-image>"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _write_results(output_dir: str | Path, result: dict[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
