from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd

from face_pipeline.classifier import DeepfakeClassifier, label_from_real_score
from face_pipeline.cropper import CropConfig, crop_face
from face_pipeline.detector import FaceDetector


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "real_vs_fake" / "real-vs-fake"
TEST_CSV = ROOT / "data" / "raw" / "test.csv"
OUTPUT_JSON = ROOT / "reports" / "phase2" / "crop_consistency.json"
OUTPUT_REPORT = ROOT / "reports" / "phase2" / "CROP_CONSISTENCY_REPORT.md"
SEED = 69


def run_crop_consistency(
    per_class: int = 25,
    seed: int = SEED,
    test_csv: Path = TEST_CSV,
    data_root: Path = DATA_ROOT,
    output_json: Path = OUTPUT_JSON,
    output_report: Path = OUTPUT_REPORT,
) -> dict[str, Any]:
    df = pd.read_csv(test_csv)
    detector = FaceDetector()
    classifier = DeepfakeClassifier()
    _ = classifier.model

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for label_name in ["real", "fake"]:
        candidates = df[df["label_str"] == label_name].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        accepted = 0
        scanned = 0
        for _, row in candidates.iterrows():
            if accepted >= per_class:
                break
            scanned += 1
            image_path = data_root / str(row["path"])
            record = compare_one_image(
                image_path=image_path,
                relative_path=str(row["path"]),
                true_label=label_name,
                detector=detector,
                classifier=classifier,
            )
            if record["status"] == "ok":
                rows.append(record)
                accepted += 1
            else:
                skipped.append(record)

        if accepted < per_class:
            raise RuntimeError(f"Only found {accepted} detectable {label_name} images after scanning {scanned} rows")

    summary = summarize(rows, skipped, per_class=per_class, seed=seed)
    payload = {"summary": summary, "records": rows, "skipped": skipped}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_report.write_text(build_markdown_report(payload), encoding="utf-8")
    return payload


def compare_one_image(
    image_path: Path,
    relative_path: str,
    true_label: str,
    detector: FaceDetector,
    classifier: DeepfakeClassifier,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        original_started = time.perf_counter()
        original_prediction = classifier.predict_image(_load_image(image_path))
        original_ms = elapsed_ms(original_started)

        detector_started = time.perf_counter()
        detections = detector.detect(image_path)
        detector_ms = elapsed_ms(detector_started)
        if not detections:
            return {
                "path": relative_path,
                "true_label": true_label,
                "status": "no_face_detected",
                "original_real_score": round(float(original_prediction["real_score"]), 6),
                "original_label": original_prediction["label"],
                "detector_time_ms": detector_ms,
                "processing_time_ms": elapsed_ms(started),
            }

        detection = detections[0]
        crop, crop_box = crop_face(image_path, detection, CropConfig())
        recrop_started = time.perf_counter()
        recrop_prediction = classifier.predict_image(crop)
        recrop_ms = elapsed_ms(recrop_started)

        original_score = float(original_prediction["real_score"])
        recrop_score = float(recrop_prediction["real_score"])
        original_class = "real" if original_score >= 0.5 else "fake"
        recrop_class = "real" if recrop_score >= 0.5 else "fake"
        return {
            "path": relative_path,
            "true_label": true_label,
            "status": "ok",
            "original_real_score": round(original_score, 6),
            "recropped_real_score": round(recrop_score, 6),
            "absolute_score_difference": round(abs(original_score - recrop_score), 6),
            "original_label": original_prediction["label"],
            "recropped_label": recrop_prediction["label"],
            "original_predicted_class": original_class,
            "recropped_predicted_class": recrop_class,
            "predicted_class_changed": original_class != recrop_class,
            "uncertainty_increased": abs(recrop_score - 0.5) < abs(original_score - 0.5),
            "face_count": len(detections),
            "bounding_box": {key: int(detection[key]) for key in ["x1", "y1", "x2", "y2"]},
            "detection_confidence": float(detection["confidence"]),
            "crop_box": crop_box,
            "crop_width": crop.width,
            "crop_height": crop.height,
            "original_classification_time_ms": original_ms,
            "detector_time_ms": detector_ms,
            "recropped_classification_time_ms": recrop_ms,
            "processing_time_ms": elapsed_ms(started),
        }
    except Exception as exc:
        return {
            "path": relative_path,
            "true_label": true_label,
            "status": "error",
            "error": repr(exc),
            "processing_time_ms": elapsed_ms(started),
        }


def summarize(rows: list[dict[str, Any]], skipped: list[dict[str, Any]], per_class: int, seed: int) -> dict[str, Any]:
    diffs = [float(row["absolute_score_difference"]) for row in rows]
    detector_times = [float(row["detector_time_ms"]) for row in rows]
    recrop_times = [float(row["recropped_classification_time_ms"]) for row in rows]
    total_times = [float(row["processing_time_ms"]) for row in rows]
    by_class: dict[str, dict[str, Any]] = {}
    for label in ["real", "fake"]:
        class_rows = [row for row in rows if row["true_label"] == label]
        class_diffs = [float(row["absolute_score_difference"]) for row in class_rows]
        by_class[label] = {
            "count": len(class_rows),
            "average_abs_score_difference": round(statistics.mean(class_diffs), 6) if class_diffs else None,
            "median_abs_score_difference": round(statistics.median(class_diffs), 6) if class_diffs else None,
            "predicted_class_changes": sum(1 for row in class_rows if row["predicted_class_changed"]),
            "uncertainty_increased": sum(1 for row in class_rows if row["uncertainty_increased"]),
        }

    return {
        "seed": seed,
        "requested_per_class": per_class,
        "records_total": len(rows),
        "records_by_class": by_class,
        "skipped_total": len(skipped),
        "average_abs_score_difference": round(statistics.mean(diffs), 6),
        "median_abs_score_difference": round(statistics.median(diffs), 6),
        "max_abs_score_difference": round(max(diffs), 6),
        "predicted_class_changes": sum(1 for row in rows if row["predicted_class_changed"]),
        "uncertainty_increased": sum(1 for row in rows if row["uncertainty_increased"]),
        "average_detector_time_ms": round(statistics.mean(detector_times), 3),
        "average_recropped_classification_time_ms": round(statistics.mean(recrop_times), 3),
        "average_processing_time_ms": round(statistics.mean(total_times), 3),
        "original_label_counts": count_values(row["original_label"] for row in rows),
        "recropped_label_counts": count_values(row["recropped_label"] for row in rows),
        "detector": "opencv_haar_frontalface_default",
        "crop_margin": 0.20,
        "crop_minimum_size": 64,
        "score_threshold_note": "Likely Fake < 0.40, Uncertain 0.40-0.60, Likely Real > 0.60",
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    class_rows = summary["records_by_class"]
    lines = [
        "# Crop Consistency Report",
        "",
        "## Method",
        "",
        f"- Dataset: `data/raw/test.csv` with images under `{DATA_ROOT}`",
        f"- Deterministic seed: `{summary['seed']}`",
        f"- Accepted images: {summary['records_total']} total ({class_rows['real']['count']} real, {class_rows['fake']['count']} fake)",
        f"- Detector: `{summary['detector']}`",
        "- Comparison: original prepared 256x256 image prediction vs. detector crop prediction.",
        "- Class-change threshold: `real_score >= 0.5` means real, otherwise fake.",
        "",
        "## Summary",
        "",
        f"- Average absolute score difference: {summary['average_abs_score_difference']:.6f}",
        f"- Median absolute score difference: {summary['median_abs_score_difference']:.6f}",
        f"- Max absolute score difference: {summary['max_abs_score_difference']:.6f}",
        f"- Predicted class changes: {summary['predicted_class_changes']} / {summary['records_total']}",
        f"- Uncertainty increased: {summary['uncertainty_increased']} / {summary['records_total']}",
        f"- Skipped images before reaching target: {summary['skipped_total']}",
        "",
        "## By Class",
        "",
        "| True class | Count | Avg abs diff | Median abs diff | Class changes | Uncertainty increased |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ["real", "fake"]:
        row = class_rows[label]
        lines.append(
            f"| {label} | {row['count']} | {row['average_abs_score_difference']:.6f} | "
            f"{row['median_abs_score_difference']:.6f} | {row['predicted_class_changes']} | "
            f"{row['uncertainty_increased']} |"
        )

    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"- Average face-detection time: {summary['average_detector_time_ms']:.3f} ms",
            f"- Average recropped classification time per face: {summary['average_recropped_classification_time_ms']:.3f} ms",
            f"- Average total processing time per image: {summary['average_processing_time_ms']:.3f} ms",
            "- Runtime: CPU TensorFlow in the existing model-audit environment.",
            "",
            "## Interpretation",
            "",
            verdict_text(summary),
            "",
            "Full row-level data is saved in `reports/phase2/crop_consistency.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def verdict_text(summary: dict[str, Any]) -> str:
    change_rate = summary["predicted_class_changes"] / summary["records_total"]
    avg_diff = summary["average_abs_score_difference"]
    if change_rate > 0.20 or avg_diff > 0.20:
        return (
            "Automatic recropping materially changes model behavior. The detector and crop settings are usable for "
            "plumbing tests, but the crop pipeline should be adjusted before API integration."
        )
    return (
        "Automatic recropping has limited effect on this deterministic subset. The pipeline is suitable for the next "
        "integration step, subject to broader user-image testing."
    )


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _load_image(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic crop-consistency evaluation.")
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_crop_consistency(per_class=args.per_class, seed=args.seed)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
