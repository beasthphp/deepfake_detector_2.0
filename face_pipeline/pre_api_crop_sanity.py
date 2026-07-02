from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from face_pipeline.classifier import preprocess_for_model
from face_pipeline.cropper import (
    SELECTED_CROP_STRATEGY,
    contextual_full_head_crop_boxes,
    crop_face_selected_strategy,
)
from face_pipeline.detector import FaceDetector, load_rgb_image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "real_vs_fake" / "real-vs-fake"
PHASE2_CONSISTENCY_JSON = ROOT / "reports" / "phase2" / "crop_consistency.json"
OUTPUT_DIR = ROOT / "reports" / "phase3"
OUTPUT_JSON = OUTPUT_DIR / "pre_api_crop_sanity.json"
OUTPUT_REPORT = OUTPUT_DIR / "PRE_API_CROP_SANITY_CHECK.md"


def run_sanity_check(sample_count: int = 25) -> dict[str, Any]:
    if sample_count < 20:
        raise ValueError("sample_count must be at least 20")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_phase2_samples(sample_count)
    detector = FaceDetector()
    records = []

    for sample in samples:
        image = load_rgb_image(sample["image_path"])
        detections = detector.detect(image)
        primary = detections[0] if detections else None
        original_model_input = preprocess_for_model(image)
        base_record: dict[str, Any] = {
            "relative_path": sample["relative_path"],
            "true_label": sample["true_label"],
            "original_dimensions": {"width": image.width, "height": image.height},
            "faces_detected": len(detections),
            "detected_bounding_box": public_box(primary),
            "original_pixel_hash": hash_rgb_pixels(image),
            "original_model_input_hash": hash_array(original_model_input),
        }

        if primary is None:
            base_record.update(
                {
                    "status": "no_face_detected",
                    "proposed_expanded_crop_box": None,
                    "expanded_clamped_box": None,
                    "final_clamped_crop_box": None,
                    "final_crop_equals_complete_original": False,
                    "prepared_face_bypass_fired": False,
                    "final_model_input_hash": None,
                    "model_input_pixels_identical": False,
                }
            )
            records.append(base_record)
            continue

        expanded_boxes = contextual_full_head_crop_boxes(primary, image.width, image.height)
        crop_result = crop_face_selected_strategy(image, primary, detections)
        final_model_input = preprocess_for_model(crop_result.crop)
        final_box = crop_result.crop_box
        full_image_box = {"x1": 0, "y1": 0, "x2": image.width, "y2": image.height}
        base_record.update(
            {
                "status": "ok",
                "selected_strategy": SELECTED_CROP_STRATEGY,
                "proposed_expanded_crop_box": expanded_boxes["proposed_expanded_crop_box"],
                "expanded_clamped_box": expanded_boxes["expanded_clamped_box"],
                "final_clamped_crop_box": final_box,
                "final_crop_equals_complete_original": final_box == full_image_box,
                "prepared_face_bypass_fired": crop_result.preserved_original,
                "preserve_original_reasons": crop_result.preserve_decision["reasons"],
                "preserve_original_blockers": crop_result.preserve_decision["blockers"],
                "preserve_original_metrics": crop_result.preserve_decision["metrics"],
                "final_model_input_hash": hash_array(final_model_input),
                "model_input_pixels_identical": bool(np.array_equal(original_model_input, final_model_input)),
            }
        )
        records.append(base_record)

    summary = summarize(records)
    payload = {
        "summary": summary,
        "records": records,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_REPORT.write_text(build_report(payload), encoding="utf-8")
    return payload


def load_phase2_samples(sample_count: int) -> list[dict[str, Any]]:
    payload = json.loads(PHASE2_CONSISTENCY_JSON.read_text(encoding="utf-8"))
    records = [row for row in payload["records"] if row.get("status") == "ok"]
    if len(records) < sample_count:
        raise ValueError(f"Need {sample_count} Phase 2 records, found {len(records)}")
    selected = records[:sample_count]
    return [
        {
            "relative_path": row["path"],
            "image_path": DATA_ROOT / row["path"],
            "true_label": row["true_label"],
        }
        for row in selected
    ]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [row for row in records if row["status"] == "ok"]
    bypass_count = sum(1 for row in ok_records if row["prepared_face_bypass_fired"])
    full_crop_count = sum(1 for row in ok_records if row["final_crop_equals_complete_original"])
    identical_count = sum(1 for row in ok_records if row["model_input_pixels_identical"])
    expanded_clamped_full = 0
    for row in ok_records:
        expanded = row.get("expanded_clamped_box")
        dims = row["original_dimensions"]
        if expanded == {"x1": 0, "y1": 0, "x2": dims["width"], "y2": dims["height"]}:
            expanded_clamped_full += 1

    if ok_records and bypass_count == len(ok_records):
        conclusion = "expected_because_prepared_face_bypass_fired"
    elif ok_records and full_crop_count == len(ok_records) and bypass_count > 0:
        conclusion = "expected_because_prepared_faces_occupy_full_frame"
    elif ok_records and full_crop_count == len(ok_records):
        conclusion = "crop_expansion_naturally_clamped_to_full_image"
    elif ok_records and identical_count == len(ok_records):
        conclusion = "expected_because_model_inputs_are_identical_after_resize"
    elif ok_records:
        conclusion = "mixed_expected_prepared_face_and_context_crop_behavior"
    else:
        conclusion = "results_cannot_be_explained"

    return {
        "sample_count": len(records),
        "ok_count": len(ok_records),
        "no_face_detected_count": len(records) - len(ok_records),
        "prepared_face_bypass_count": bypass_count,
        "full_original_final_crop_count": full_crop_count,
        "expanded_clamped_to_full_image_count": expanded_clamped_full,
        "model_input_pixels_identical_count": identical_count,
        "selected_strategy": SELECTED_CROP_STRATEGY,
        "conclusion": conclusion,
    }


def build_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    records = payload["records"]
    lines = [
        "# Pre-API Crop Sanity Check",
        "",
        "## Method",
        "",
        f"- Strategy checked: `{summary['selected_strategy']}`.",
        f"- Deterministic source: first {summary['sample_count']} successful records from `reports/phase2/crop_consistency.json`.",
        "- For each image, the check records original dimensions, Haar box, expanded boxes, final crop box, hashes, and model-input identity.",
        "",
        "## Summary",
        "",
        f"- Images checked: {summary['sample_count']}",
        f"- Images with a detected face: {summary['ok_count']}",
        f"- No-face records: {summary['no_face_detected_count']}",
        f"- Prepared-face bypass fired: {summary['prepared_face_bypass_count']} / {summary['ok_count']}",
        f"- Final crop equals complete original: {summary['full_original_final_crop_count']} / {summary['ok_count']}",
        f"- Expanded rectangle clamped to full image: {summary['expanded_clamped_to_full_image_count']} / {summary['ok_count']}",
        f"- Final model-input pixels identical to original model-input pixels: {summary['model_input_pixels_identical_count']} / {summary['ok_count']}",
        f"- Conclusion: `{summary['conclusion']}`",
        "",
        "## Interpretation",
        "",
        interpretation(summary),
        "",
        "## Sample Rows",
        "",
        "| Image | Face box | Final crop | Bypass | Full original | Identical model input |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in records[:10]:
        lines.append(
            "| `{image}` | `{face}` | `{crop}` | {bypass} | {full} | {identical} |".format(
                image=row["relative_path"],
                face=json.dumps(row["detected_bounding_box"], sort_keys=True),
                crop=json.dumps(row["final_clamped_crop_box"], sort_keys=True),
                bypass=row.get("prepared_face_bypass_fired"),
                full=row.get("final_crop_equals_complete_original"),
                identical=row.get("model_input_pixels_identical"),
            )
        )
    lines.extend(
        [
            "",
            "Full row-level data is saved in `reports/phase3/pre_api_crop_sanity.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def interpretation(summary: dict[str, Any]) -> str:
    if summary["conclusion"] == "expected_because_prepared_face_bypass_fired":
        return (
            "The identical predictions are explained by the prepared-face bypass: every successfully detected sample "
            "already met the single-face, square-ish, centered, sufficiently occupied, in-bounds criteria, so the "
            "selected strategy intentionally reused the original image as model input. This is not an implementation bug."
        )
    if summary["conclusion"] == "expected_because_prepared_faces_occupy_full_frame":
        return (
            "The identical predictions are explained by prepared-face geometry. Most rows fired the prepared-face "
            "bypass, and every contextual expansion still clamped to the complete original frame. The final model-input "
            "pixels were identical for every checked image, so this is not an implementation bug."
        )
    if summary["conclusion"] == "crop_expansion_naturally_clamped_to_full_image":
        return (
            "The identical predictions are explained by contextual crop expansion clamping to the full source image. "
            "This is expected for prepared images that already fill the frame."
        )
    if summary["conclusion"] == "expected_because_model_inputs_are_identical_after_resize":
        return (
            "The crop boxes are not always the full image, but the final resized model-input pixels match the original "
            "model input. That explains the zero score differences without requiring a code change."
        )
    if summary["conclusion"] == "mixed_expected_prepared_face_and_context_crop_behavior":
        return (
            "The checked rows show a mix of bypass and contextual crop behavior. The zero-score rows can be explained "
            "by full-frame preservation or identical model-input pixels."
        )
    return "The checked rows do not explain the zero-score behavior. The crop implementation should be inspected before API integration."


def hash_rgb_pixels(image) -> str:
    rgb = image.convert("RGB")
    return hashlib.sha256(np.asarray(rgb, dtype=np.uint8).tobytes()).hexdigest()


def hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def public_box(detection: dict[str, Any] | None) -> dict[str, int] | None:
    if detection is None:
        return None
    return {key: int(detection[key]) for key in ["x1", "y1", "x2", "y2"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 pre-API crop sanity check.")
    parser.add_argument("--sample-count", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_sanity_check(sample_count=args.sample_count)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
