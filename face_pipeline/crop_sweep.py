from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from face_pipeline.classifier import DeepfakeClassifier, label_from_real_score, preprocess_for_model
from face_pipeline.detector import FaceDetector, load_rgb_image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "real_vs_fake" / "real-vs-fake"
TEST_CSV = ROOT / "data" / "raw" / "test.csv"
VALID_CSV = ROOT / "data" / "raw" / "valid.csv"
PHASE2_CONSISTENCY_JSON = ROOT / "reports" / "phase2" / "crop_consistency.json"
OUTPUT_DIR = ROOT / "reports" / "phase2" / "crop_sweep"
SAMPLES_DIR = ROOT / "phase2_samples" / "crop_sweep"
REPORT_PATH = ROOT / "reports" / "phase2" / "PHASE2_CROP_SWEEP_REPORT.md"
SEED = 69
MODEL_INPUT_SIZE = (256, 256)


@dataclass(frozen=True)
class PreserveConfig:
    square_tolerance: float = 0.10
    centered_offset_threshold: float = 0.12
    face_occupancy_threshold: float = 0.30
    boundary_fraction: float = 0.015


@dataclass(frozen=True)
class CropStrategy:
    name: str
    family: str
    description: str
    margin: float | None = None
    expansion: dict[str, float] | None = None
    padding_method: str | None = None
    fallback: "CropStrategy | None" = field(default=None, compare=False)

    def config(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "margin": self.margin,
            "expansion": self.expansion,
            "padding_method": self.padding_method,
            "fallback": self.fallback.name if self.fallback else None,
        }


@dataclass
class ImageRecord:
    relative_path: str
    image_path: Path
    true_label: str
    split: str
    image_width: int
    image_height: int
    detections: list[dict[str, Any]]
    detector_time_ms: float

    @property
    def primary_detection(self) -> dict[str, Any] | None:
        return self.detections[0] if self.detections else None


def should_preserve_original(
    image_size: tuple[int, int],
    detections: list[dict[str, Any]],
    config: PreserveConfig | None = None,
) -> dict[str, Any]:
    """Decide whether an image already appears to be a prepared face crop."""

    preserve_config = config or PreserveConfig()
    width, height = image_size
    reasons: list[str] = []
    blockers: list[str] = []

    if len(detections) == 1:
        reasons.append("single_face")
    elif not detections:
        blockers.append("no_face_detected")
    else:
        blockers.append("multiple_faces_detected")

    aspect_ratio = width / height if height else 0.0
    if _close_to_square(width, height, preserve_config.square_tolerance):
        reasons.append("square_image")
    else:
        blockers.append("image_not_square")

    if width == 256 and height == 256:
        reasons.append("image_is_256x256")

    metrics: dict[str, Any] = {
        "image_aspect_ratio": round(aspect_ratio, 6),
        "face_occupancy_ratio": None,
        "face_center_offset_x": None,
        "face_center_offset_y": None,
        "face_center_offset_max": None,
    }

    if detections:
        detection = detections[0]
        geom = face_geometry((width, height), detection)
        metrics.update(geom)

        if geom["face_center_offset_max"] <= preserve_config.centered_offset_threshold:
            reasons.append("face_is_centered")
        else:
            blockers.append("face_not_centered")

        if geom["face_occupancy_ratio"] >= preserve_config.face_occupancy_threshold:
            reasons.append("face_occupancy_above_threshold")
        else:
            blockers.append("face_occupancy_below_threshold")

        boundary = max(1, int(round(min(width, height) * preserve_config.boundary_fraction)))
        if (
            detection["x1"] > boundary
            and detection["y1"] > boundary
            and detection["x2"] < width - boundary
            and detection["y2"] < height - boundary
        ):
            reasons.append("bounding_box_inside_image")
        else:
            blockers.append("bounding_box_touches_or_nears_boundary")

    required = {
        "single_face",
        "square_image",
        "face_is_centered",
        "face_occupancy_above_threshold",
        "bounding_box_inside_image",
    }
    preserve_original = required.issubset(set(reasons))
    return {
        "preserve_original": preserve_original,
        "reasons": reasons,
        "blockers": blockers,
        "metrics": metrics,
        "config": {
            "square_tolerance": preserve_config.square_tolerance,
            "centered_offset_threshold": preserve_config.centered_offset_threshold,
            "face_occupancy_threshold": preserve_config.face_occupancy_threshold,
            "boundary_fraction": preserve_config.boundary_fraction,
        },
    }


def base_strategies() -> list[CropStrategy]:
    full_head = {"left": 0.40, "right": 0.40, "top": 0.70, "bottom": 0.35}
    strategies = [
        CropStrategy("original_reference", "original", "Use the complete original image without recropping."),
        CropStrategy("square_m20_current", "square_margin", "Current Phase 2 square crop with 20% margin.", margin=0.20),
    ]
    for margin in [0.30, 0.40, 0.50, 0.65, 0.80]:
        strategies.append(
            CropStrategy(
                f"square_m{int(margin * 100):02d}",
                "square_margin",
                f"Larger square crop with {int(margin * 100)}% margin.",
                margin=margin,
            )
        )
    strategies.extend(
        [
            CropStrategy(
                "context_full_head_l40_r40_t70_b35_square",
                "context_square",
                "Full-head contextual square crop with 40/40/70/35% left/right/top/bottom expansion.",
                expansion=full_head,
            ),
            CropStrategy(
                "context_l40_r40_t70_b35_stretch",
                "context_stretch",
                "Expanded non-square context crop resized directly to 256x256, introducing geometric stretching.",
                expansion=full_head,
            ),
            CropStrategy(
                "padded_context_black",
                "context_padded",
                "Expanded context crop resized with aspect preservation and black padding.",
                expansion=full_head,
                padding_method="black",
            ),
            CropStrategy(
                "padded_context_reflect",
                "context_padded",
                "Expanded context crop resized with aspect preservation and reflected-edge padding.",
                expansion=full_head,
                padding_method="reflect",
            ),
            CropStrategy(
                "padded_context_edge",
                "context_padded",
                "Expanded context crop resized with aspect preservation and replicated-edge padding.",
                expansion=full_head,
                padding_method="edge",
            ),
        ]
    )
    return strategies


def run_all() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    fifty_payload = run_fifty_image_sweep()
    top_two = select_expanded_strategies(fifty_payload["ranking"], fifty_payload["strategies"])
    expanded_payload = run_expanded_validation(top_two)
    failure_payload = save_failure_examples(
        rows=fifty_payload["rows"],
        top_strategy_names=[row["strategy_name"] for row in fifty_payload["ranking"][:3]],
    )
    report_text = build_report(fifty_payload, expanded_payload, failure_payload, top_two)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    return {
        "fifty_image_sweep": fifty_payload["summary"],
        "expanded_validation": expanded_payload["summary"],
        "failure_examples": failure_payload,
        "report": str(REPORT_PATH),
    }


def run_fifty_image_sweep() -> dict[str, Any]:
    samples = load_previous_consistency_sample()
    records = prepare_records(samples)
    classifier = DeepfakeClassifier()
    _ = classifier.model
    original_predictions = predict_originals(records, classifier)

    strategies_by_name = {strategy.name: strategy for strategy in base_strategies()}
    rows: list[dict[str, Any]] = []
    for strategy in strategies_by_name.values():
        rows.extend(evaluate_strategy(records, strategy, classifier, original_predictions))

    summaries = summarize_rows(rows, sample_count=len(records))
    non_reference_rank = rank_summaries(
        [summary for summary in summaries if summary["strategy_name"] != "original_reference"]
    )
    context_rank = rank_summaries(
        [
            summary
            for summary in summaries
            if summary["strategy_family"] in {"context_square", "context_stretch", "context_padded"}
        ]
    )
    best_fallback = strategies_by_name[context_rank[0]["strategy_name"]]
    conditional = CropStrategy(
        name=f"preserve_or_{best_fallback.name}",
        family="prepared_face_conditional",
        description=(
            "Use the original image when the prepared-face bypass fires; otherwise use "
            f"`{best_fallback.name}`."
        ),
        fallback=best_fallback,
    )
    strategies_by_name[conditional.name] = conditional
    rows.extend(evaluate_strategy(records, conditional, classifier, original_predictions))

    summaries = summarize_rows(rows, sample_count=len(records))
    ranking = rank_summaries(summaries)
    write_csv(OUTPUT_DIR / "crop_sweep_predictions.csv", rows)
    write_csv(OUTPUT_DIR / "crop_sweep_ranking.csv", ranking)
    payload = {
        "summary": {
            "split": "test",
            "sample_count": len(records),
            "seed": SEED,
            "source": str(PHASE2_CONSISTENCY_JSON),
            "best_non_reference_strategy": non_reference_rank[0]["strategy_name"],
            "best_fallback_strategy": best_fallback.name,
        },
        "strategies": {name: strategy_to_json(strategy) for name, strategy in strategies_by_name.items()},
        "summaries": summaries,
        "ranking": ranking,
        "rows": rows,
    }
    (OUTPUT_DIR / "crop_sweep_summary.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return payload


def run_expanded_validation(strategy_defs: list[dict[str, Any]]) -> dict[str, Any]:
    samples = load_split_sample(VALID_CSV, split="valid", per_class=100, seed=SEED)
    records = prepare_records(samples)
    classifier = DeepfakeClassifier()
    _ = classifier.model
    original_predictions = predict_originals(records, classifier)

    available = {strategy.name: strategy for strategy in base_strategies()}
    strategies = [strategy_from_json(definition, available) for definition in strategy_defs]
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        rows.extend(evaluate_strategy(records, strategy, classifier, original_predictions))

    summaries = summarize_rows(rows, sample_count=len(records))
    ranking = rank_summaries(summaries)
    payload = {
        "summary": {
            "split": "valid",
            "sample_count": len(records),
            "seed": SEED,
            "strategies": [strategy.name for strategy in strategies],
        },
        "strategies": {strategy.name: strategy_to_json(strategy) for strategy in strategies},
        "summaries": summaries,
        "ranking": ranking,
        "rows": rows,
    }
    write_csv(OUTPUT_DIR / "expanded_predictions.csv", rows)
    (OUTPUT_DIR / "expanded_results.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return payload


def load_previous_consistency_sample() -> list[dict[str, Any]]:
    if not PHASE2_CONSISTENCY_JSON.exists():
        raise FileNotFoundError(f"Missing previous consistency file: {PHASE2_CONSISTENCY_JSON}")
    payload = json.loads(PHASE2_CONSISTENCY_JSON.read_text(encoding="utf-8"))
    records = [row for row in payload["records"] if row.get("status") == "ok"]
    if len(records) != 50:
        raise ValueError(f"Expected exactly 50 previous consistency records, found {len(records)}")
    return [
        {
            "relative_path": row["path"],
            "image_path": DATA_ROOT / row["path"],
            "true_label": row["true_label"],
            "split": "test",
        }
        for row in records
    ]


def load_split_sample(csv_path: Path, split: str, per_class: int, seed: int) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    samples: list[dict[str, Any]] = []
    for label_name in ["real", "fake"]:
        class_df = df[df["label_str"] == label_name].sample(frac=1.0, random_state=seed).head(per_class)
        if len(class_df) < per_class:
            raise ValueError(f"Need {per_class} {label_name} rows, found {len(class_df)}")
        for _, row in class_df.iterrows():
            samples.append(
                {
                    "relative_path": str(row["path"]),
                    "image_path": DATA_ROOT / str(row["path"]),
                    "true_label": label_name,
                    "split": split,
                }
            )
    return samples


def prepare_records(samples: list[dict[str, Any]]) -> list[ImageRecord]:
    detector = FaceDetector()
    records: list[ImageRecord] = []
    for sample in samples:
        image = load_rgb_image(sample["image_path"])
        started = time.perf_counter()
        detections = detector.detect(image)
        detector_time_ms = elapsed_ms(started)
        records.append(
            ImageRecord(
                relative_path=sample["relative_path"],
                image_path=Path(sample["image_path"]),
                true_label=sample["true_label"],
                split=sample["split"],
                image_width=image.width,
                image_height=image.height,
                detections=detections,
                detector_time_ms=detector_time_ms,
            )
        )
    return records


def predict_originals(records: list[ImageRecord], classifier: DeepfakeClassifier) -> dict[str, dict[str, Any]]:
    images = [make_original_input(load_rgb_image(record.image_path)) for record in records]
    scores = predict_scores(classifier, images)
    predictions: dict[str, dict[str, Any]] = {}
    for record, score in zip(records, scores):
        label = label_from_real_score(score)
        predictions[record.relative_path] = {
            "score": score,
            "label": label,
            "class": class_from_score(score),
        }
    return predictions


def evaluate_strategy(
    records: list[ImageRecord],
    strategy: CropStrategy,
    classifier: DeepfakeClassifier,
    original_predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    model_inputs: list[Image.Image] = []
    prep_started = time.perf_counter()
    for record in records:
        original = load_rgb_image(record.image_path)
        preserve_decision = should_preserve_original((record.image_width, record.image_height), record.detections)
        base_row = base_result_row(record, strategy, original_predictions[record.relative_path], preserve_decision)
        try:
            prepared_input = prepare_strategy_input(original, record.primary_detection, strategy, preserve_decision)
            base_row.update(prepared_input["row"])
            if prepared_input["status"] == "ok":
                model_inputs.append(prepared_input["model_input"])
                prepared.append({"row": base_row, "status": "ok"})
            else:
                prepared.append({"row": base_row, "status": prepared_input["status"]})
        except Exception as exc:
            base_row.update({"status": "crop_error", "error": repr(exc)})
            prepared.append({"row": base_row, "status": "crop_error"})

    prep_ms_per_image = elapsed_ms(prep_started) / max(len(records), 1)
    prediction_started = time.perf_counter()
    scores = predict_scores(classifier, model_inputs) if model_inputs else []
    predict_ms_per_ok = elapsed_ms(prediction_started) / max(len(model_inputs), 1) if model_inputs else 0.0

    score_iter = iter(scores)
    rows: list[dict[str, Any]] = []
    for item in prepared:
        row = item["row"]
        if item["status"] != "ok":
            row.setdefault("status", item["status"])
            row.update(failed_prediction_fields())
            rows.append(row)
            continue

        cropped_score = next(score_iter)
        original_score = float(row["original_model_score"])
        cropped_label = label_from_real_score(cropped_score)
        cropped_class = class_from_score(cropped_score)
        original_class = row["original_predicted_class"]
        true_label = row["true_label"]
        class_changed = cropped_class != original_class
        row.update(
            {
                "status": "ok",
                "cropped_model_score": round(cropped_score, 6),
                "absolute_score_difference": round(abs(original_score - cropped_score), 6),
                "cropped_label": cropped_label,
                "cropped_predicted_class": cropped_class,
                "class_changed": class_changed,
                "fake_to_real_flip": bool(true_label == "fake" and original_class == "fake" and cropped_class == "real"),
                "real_to_fake_flip": bool(true_label == "real" and original_class == "real" and cropped_class == "fake"),
                "became_uncertain": bool(cropped_label == "Uncertain" and row["original_label"] != "Uncertain"),
                "correct_after_crop": bool(cropped_class == true_label),
                "preparation_time_ms": round(prep_ms_per_image, 3),
                "classification_time_ms": round(predict_ms_per_ok, 3),
                "processing_time_ms": round(record.detector_time_ms + prep_ms_per_image + predict_ms_per_ok, 3),
            }
        )
        rows.append(row)
    return rows


def prepare_strategy_input(
    image: Image.Image,
    detection: dict[str, Any] | None,
    strategy: CropStrategy,
    preserve_decision: dict[str, Any],
) -> dict[str, Any]:
    if strategy.family == "original":
        crop_box = {"x1": 0, "y1": 0, "x2": image.width, "y2": image.height}
        model_input = make_original_input(image)
        return {
            "status": "ok",
            "model_input": model_input,
            "row": crop_row_fields(
                crop_box,
                image.width,
                image.height,
                strategy,
                resize_method="nearest",
                padding_method="none",
                geometric_stretching=image.size != MODEL_INPUT_SIZE,
                used_preserve_original=True,
            ),
        }

    if strategy.family == "prepared_face_conditional":
        if preserve_decision["preserve_original"]:
            crop_box = {"x1": 0, "y1": 0, "x2": image.width, "y2": image.height}
            model_input = make_original_input(image)
            return {
                "status": "ok",
                "model_input": model_input,
                "row": crop_row_fields(
                    crop_box,
                    image.width,
                    image.height,
                    strategy,
                    resize_method="nearest",
                    padding_method="none",
                    geometric_stretching=image.size != MODEL_INPUT_SIZE,
                    used_preserve_original=True,
                ),
            }
        if strategy.fallback is None:
            return {"status": "crop_error", "row": {"error": "conditional strategy missing fallback"}}
        fallback_prepared = prepare_strategy_input(image, detection, strategy.fallback, preserve_decision)
        fallback_prepared["row"]["used_preserve_original"] = False
        fallback_prepared["row"]["fallback_strategy"] = strategy.fallback.name
        return fallback_prepared

    if detection is None:
        return {"status": "no_face_detected", "row": {}}

    if strategy.family == "square_margin":
        crop_box = square_crop_box(detection, image.width, image.height, strategy.margin or 0.0)
        crop = crop_image(image, crop_box)
        model_input = resize_nearest(crop)
        return {
            "status": "ok",
            "model_input": model_input,
            "row": crop_row_fields(crop_box, crop.width, crop.height, strategy, "nearest", "none", False),
        }

    if strategy.family == "context_square":
        expanded = expand_detection_box(detection, image.width, image.height, strategy.expansion or {})
        crop_box = square_from_rect(expanded, image.width, image.height)
        crop = crop_image(image, crop_box)
        model_input = resize_nearest(crop)
        return {
            "status": "ok",
            "model_input": model_input,
            "row": crop_row_fields(crop_box, crop.width, crop.height, strategy, "nearest", "none", False),
        }

    if strategy.family == "context_stretch":
        crop_box = expand_detection_box(detection, image.width, image.height, strategy.expansion or {})
        crop = crop_image(image, crop_box)
        model_input = resize_nearest(crop)
        return {
            "status": "ok",
            "model_input": model_input,
            "row": crop_row_fields(crop_box, crop.width, crop.height, strategy, "nearest", "none", True),
        }

    if strategy.family == "context_padded":
        crop_box = expand_detection_box(detection, image.width, image.height, strategy.expansion or {})
        crop = crop_image(image, crop_box)
        model_input, pad_info = resize_with_padding(crop, strategy.padding_method or "edge")
        row = crop_row_fields(
            crop_box,
            crop.width,
            crop.height,
            strategy,
            resize_method="nearest_aspect_preserving",
            padding_method=strategy.padding_method or "edge",
            geometric_stretching=False,
        )
        row["padding_pixels"] = json.dumps(pad_info, sort_keys=True)
        return {"status": "ok", "model_input": model_input, "row": row}

    raise ValueError(f"Unknown crop strategy family: {strategy.family}")


def base_result_row(
    record: ImageRecord,
    strategy: CropStrategy,
    original_prediction: dict[str, Any],
    preserve_decision: dict[str, Any],
) -> dict[str, Any]:
    detection = record.primary_detection
    geom = face_geometry((record.image_width, record.image_height), detection) if detection else empty_geometry()
    return {
        "split": record.split,
        "relative_path": record.relative_path,
        "image_path": str(record.image_path),
        "image_name": Path(record.relative_path).name,
        "true_label": record.true_label,
        "strategy_name": strategy.name,
        "strategy_family": strategy.family,
        "strategy_description": strategy.description,
        "original_image_width": record.image_width,
        "original_image_height": record.image_height,
        "face_count": len(record.detections),
        "detected_face_box": json.dumps(public_box(detection), sort_keys=True) if detection else "",
        "detection_confidence": detection.get("confidence") if detection else "",
        "detector_time_ms": record.detector_time_ms,
        "face_occupancy_ratio": geom["face_occupancy_ratio"],
        "face_center_offset_x": geom["face_center_offset_x"],
        "face_center_offset_y": geom["face_center_offset_y"],
        "face_center_offset_max": geom["face_center_offset_max"],
        "preserve_original_decision": bool(preserve_decision["preserve_original"]),
        "preserve_original_reasons": json.dumps(preserve_decision["reasons"]),
        "preserve_original_blockers": json.dumps(preserve_decision["blockers"]),
        "used_preserve_original": False,
        "fallback_strategy": "",
        "original_model_score": round(float(original_prediction["score"]), 6),
        "original_label": original_prediction["label"],
        "original_predicted_class": original_prediction["class"],
        "crop_margin_configuration": json.dumps(strategy.config(), sort_keys=True),
    }


def crop_row_fields(
    crop_box: dict[str, int],
    crop_width: int,
    crop_height: int,
    strategy: CropStrategy,
    resize_method: str,
    padding_method: str,
    geometric_stretching: bool,
    used_preserve_original: bool = False,
) -> dict[str, Any]:
    return {
        "crop_box": json.dumps(crop_box, sort_keys=True),
        "crop_width_before_resize": int(crop_width),
        "crop_height_before_resize": int(crop_height),
        "resize_method": resize_method,
        "padding_method": padding_method,
        "padding_pixels": "",
        "geometric_stretching": geometric_stretching,
        "used_preserve_original": used_preserve_original,
        "crop_margin_configuration": json.dumps(strategy.config(), sort_keys=True),
    }


def failed_prediction_fields() -> dict[str, Any]:
    return {
        "cropped_model_score": "",
        "absolute_score_difference": "",
        "cropped_label": "",
        "cropped_predicted_class": "",
        "class_changed": "",
        "fake_to_real_flip": "",
        "real_to_fake_flip": "",
        "became_uncertain": "",
        "correct_after_crop": "",
        "preparation_time_ms": "",
        "classification_time_ms": "",
        "processing_time_ms": "",
    }


def summarize_rows(rows: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for strategy_name in sorted({row["strategy_name"] for row in rows}):
        strategy_rows = [row for row in rows if row["strategy_name"] == strategy_name]
        ok_rows = [row for row in strategy_rows if row.get("status") == "ok"]
        diffs = [float(row["absolute_score_difference"]) for row in ok_rows]
        true_fake = [row for row in ok_rows if row["true_label"] == "fake"]
        true_real = [row for row in ok_rows if row["true_label"] == "real"]
        failed_detection_count = sum(1 for row in strategy_rows if row.get("status") == "no_face_detected")
        crop_failure_count = sum(1 for row in strategy_rows if row.get("status") not in {"ok", "no_face_detected"})
        fake_recall = safe_div(sum(1 for row in true_fake if row["cropped_predicted_class"] == "fake"), len(true_fake))
        real_recall = safe_div(sum(1 for row in true_real if row["cropped_predicted_class"] == "real"), len(true_real))
        summary = {
            "strategy_name": strategy_name,
            "strategy_family": strategy_rows[0]["strategy_family"],
            "evaluated_image_count": len(ok_rows),
            "sample_count": sample_count,
            "failed_detection_count": failed_detection_count,
            "crop_failure_count": crop_failure_count,
            "failure_rate": round((failed_detection_count + crop_failure_count) / max(sample_count, 1), 6),
            "average_abs_score_difference": round(float(statistics.mean(diffs)), 6) if diffs else None,
            "median_abs_score_difference": round(float(statistics.median(diffs)), 6) if diffs else None,
            "p90_abs_score_difference": round(float(np.percentile(diffs, 90)), 6) if diffs else None,
            "max_abs_score_difference": round(float(max(diffs)), 6) if diffs else None,
            "total_class_changes": sum(1 for row in ok_rows if bool(row["class_changed"])),
            "fake_to_real_flips": sum(1 for row in ok_rows if bool(row["fake_to_real_flip"])),
            "real_to_fake_flips": sum(1 for row in ok_rows if bool(row["real_to_fake_flip"])),
            "uncertain_results": sum(1 for row in ok_rows if row["cropped_label"] == "Uncertain"),
            "became_uncertain": sum(1 for row in ok_rows if bool(row["became_uncertain"])),
            "accuracy_after_crop": round(safe_div(sum(1 for row in ok_rows if bool(row["correct_after_crop"])), len(ok_rows)), 6),
            "fake_recall_after_crop": round(fake_recall, 6),
            "real_recall_after_crop": round(real_recall, 6),
            "fake_to_real_flip_rate": round(safe_div(sum(1 for row in ok_rows if bool(row["fake_to_real_flip"])), len(true_fake)), 6),
            "total_class_change_rate": round(safe_div(sum(1 for row in ok_rows if bool(row["class_changed"])), len(ok_rows)), 6),
            "preserved_original_count": sum(1 for row in ok_rows if bool(row["used_preserve_original"])),
            "average_processing_time_ms": round(
                float(statistics.mean([float(row["processing_time_ms"]) for row in ok_rows])), 3
            )
            if ok_rows
            else None,
        }
        summaries.append(summary)
    return summaries


def rank_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        summaries,
        key=lambda row: (
            row["fake_to_real_flips"],
            row["total_class_changes"],
            -row["fake_recall_after_crop"],
            row["median_abs_score_difference"] if row["median_abs_score_difference"] is not None else math.inf,
            row["p90_abs_score_difference"] if row["p90_abs_score_difference"] is not None else math.inf,
            row["max_abs_score_difference"] if row["max_abs_score_difference"] is not None else math.inf,
            strategy_artifact_penalty(row["strategy_name"], row["strategy_family"]),
            row["failure_rate"],
            row["average_processing_time_ms"] if row["average_processing_time_ms"] is not None else math.inf,
        ),
    )
    output = []
    for rank, row in enumerate(ranked, start=1):
        ranked_row = dict(row)
        ranked_row["rank"] = rank
        output.append(ranked_row)
    return output


def select_expanded_strategies(ranking: list[dict[str, Any]], strategies: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_names: set[str] = set()
    for row in ranking:
        if row["strategy_name"] == "original_reference" or row["strategy_family"] == "prepared_face_conditional":
            continue
        selected.append(strategies[row["strategy_name"]])
        selected_names.add(row["strategy_name"])
        break

    for row in ranking:
        if row["strategy_family"] != "prepared_face_conditional":
            continue
        if row["strategy_name"] in selected_names:
            continue
        selected.append(strategies[row["strategy_name"]])
        selected_names.add(row["strategy_name"])
        break

    for row in ranking:
        if len(selected) >= 2:
            break
        if row["strategy_name"] == "original_reference" or row["strategy_name"] in selected_names:
            continue
        selected.append(strategies[row["strategy_name"]])
        selected_names.add(row["strategy_name"])

    if len(selected) < 2:
        raise RuntimeError("Could not select two non-reference strategies for expanded validation")
    return selected


def save_failure_examples(rows: list[dict[str, Any]], top_strategy_names: list[str]) -> dict[str, Any]:
    saved: list[dict[str, Any]] = []
    by_strategy: dict[str, int] = {}
    strategies = {strategy.name: strategy for strategy in base_strategies()}
    for row in rows:
        if row["strategy_name"] not in top_strategy_names or not row.get("fake_to_real_flip"):
            continue
        strategy = strategies.get(row["strategy_name"])
        if strategy is None:
            continue
        image_path = Path(row["image_path"])
        image = load_rgb_image(image_path)
        detection = json.loads(row["detected_face_box"]) if row["detected_face_box"] else None
        preserve_decision = should_preserve_original((image.width, image.height), [detection] if detection else [])
        prepared = prepare_strategy_input(image, detection, strategy, preserve_decision)
        if prepared["status"] != "ok":
            continue

        example_dir = SAMPLES_DIR / row["strategy_name"] / sanitize_name(row["relative_path"])
        example_dir.mkdir(parents=True, exist_ok=True)
        image.save(example_dir / "original.jpg")
        draw_detected_box(image, detection).save(example_dir / "detected_box.jpg")
        crop_box = json.loads(row["crop_box"]) if row.get("crop_box") else {"x1": 0, "y1": 0, "x2": image.width, "y2": image.height}
        crop_before = crop_image(image, crop_box)
        crop_before.save(example_dir / "crop_before_resize.jpg")
        prepared["model_input"].save(example_dir / "model_input.jpg")
        causes = infer_failure_causes(row)
        result = dict(row)
        result["failure_cause_assessment"] = causes
        (example_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
        saved.append({"strategy_name": row["strategy_name"], "relative_path": row["relative_path"], "causes": causes})
        by_strategy[row["strategy_name"]] = by_strategy.get(row["strategy_name"], 0) + 1
    return {"saved_examples": saved, "counts_by_strategy": by_strategy, "top_strategy_names": top_strategy_names}


def infer_failure_causes(row: dict[str, Any]) -> list[str]:
    causes: list[str] = []
    if not row.get("crop_box") or not row.get("detected_face_box"):
        return ["insufficient_geometry_for_assessment"]
    crop_box = json.loads(row["crop_box"])
    face_box = json.loads(row["detected_face_box"])
    crop_w = max(1, crop_box["x2"] - crop_box["x1"])
    crop_h = max(1, crop_box["y2"] - crop_box["y1"])
    face_w = max(1, face_box["x2"] - face_box["x1"])
    face_h = max(1, face_box["y2"] - face_box["y1"])
    face_crop_occupancy = (face_w * face_h) / max(1, crop_w * crop_h)
    top_margin = face_box["y1"] - crop_box["y1"]
    bottom_margin = crop_box["y2"] - face_box["y2"]
    left_margin = face_box["x1"] - crop_box["x1"]
    right_margin = crop_box["x2"] - face_box["x2"]

    if face_crop_occupancy > 0.55:
        causes.append("crop_too_tight")
    if face_crop_occupancy < 0.18:
        causes.append("crop_too_loose_or_excessive_background")
    if top_margin < 0.25 * face_h:
        causes.append("hair_or_forehead_likely_removed")
    if bottom_margin < 0.15 * face_h:
        causes.append("jaw_or_chin_likely_removed")
    if min(top_margin, bottom_margin, left_margin, right_margin) <= 2:
        causes.append("face_or_crop_near_image_boundary")
    if row.get("geometric_stretching") in {True, "True", "true"}:
        causes.append("resize_distortion")
    if row.get("padding_method") in {"black", "reflect", "edge"} and row.get("padding_pixels"):
        pads = json.loads(row["padding_pixels"])
        if max(pads.values()) >= 24:
            causes.append("padding_artifacts_possible")
    if min(crop_w, crop_h) < 96:
        causes.append("low_resolution_face_crop")
    if not causes:
        causes.append("model_sensitivity_despite_reasonable_crop")
    return causes


def build_report(
    fifty_payload: dict[str, Any],
    expanded_payload: dict[str, Any],
    failure_payload: dict[str, Any],
    top_two: list[dict[str, Any]],
) -> str:
    fifty_ranking = fifty_payload["ranking"]
    expanded_ranking = expanded_payload["ranking"]
    selected = choose_selected_candidate(expanded_ranking)
    fifty_top = fifty_ranking[0]
    best_fallback = fifty_payload["summary"]["best_fallback_strategy"]
    gate = integration_gate(selected)
    detector_assessment = detector_verdict(fifty_payload, failure_payload)
    verdict = phase_verdict(selected, gate, detector_assessment)

    lines = [
        "# Phase 2.1 Crop Sweep Report",
        "",
        "## A. Strategies Tested",
        "",
        "- `original_reference`: complete original image, reference only.",
        "- `square_m20_current`: Phase 2 baseline square crop with 20% margin.",
        "- `square_m30`, `square_m40`, `square_m50`, `square_m65`, `square_m80`: larger square margins.",
        "- `context_full_head_l40_r40_t70_b35_square`: contextual square crop with extra top area.",
        "- `context_l40_r40_t70_b35_stretch`: contextual non-square crop stretched to 256x256.",
        "- `padded_context_black`, `padded_context_reflect`, `padded_context_edge`: contextual crop resized with aspect preservation and padding.",
        f"- `preserve_or_{best_fallback}`: prepared-face bypass, otherwise `{best_fallback}`.",
        "",
        "The prepared-face bypass requires a single detected face, near-square image, centered face, high face occupancy, and a box inside image boundaries. A 256x256 image is only one supporting reason, never sufficient by itself.",
        "",
        "## B. Fifty-Image Sweep",
        "",
        "The first sweep reused the exact 50 records from `reports/phase2/crop_consistency.json`.",
        "",
        ranking_table(fifty_ranking),
        "",
        f"Best non-reference fallback from the 50-image sweep: `{best_fallback}`.",
        f"Top ranked strategy including the reference: `{fifty_top['strategy_name']}`.",
        "",
        "Row-level predictions are saved in `reports/phase2/crop_sweep/crop_sweep_predictions.csv`.",
        "",
        "## C. Expanded Validation",
        "",
        "The top contextual crop and the prepared-face conditional candidate from the 50-image sweep were evaluated on 100 real and 100 fake validation images selected with seed 69.",
        "",
        ranking_table(expanded_ranking),
        "",
        "Expanded row-level predictions are saved in `reports/phase2/crop_sweep/expanded_predictions.csv`.",
        "",
        "## D. Failure Analysis",
        "",
        failure_summary_text(failure_payload),
        "",
        "Failure examples are saved under `phase2_samples/crop_sweep/` with original, detected-box, crop, model-input, and JSON result files.",
        "",
        "## E. Detector Assessment",
        "",
        detector_assessment,
        "",
        "## F. Selected Production Candidate",
        "",
        f"Selected candidate: `{selected['strategy_name']}`.",
        "",
        selected_strategy_text(top_two, selected),
        "",
        "Internal MVP gate on expanded validation:",
        "",
        f"- Fake-to-real flip rate: {selected['fake_to_real_flip_rate']:.6f}",
        f"- Total class-change rate: {selected['total_class_change_rate']:.6f}",
        f"- Detection/crop failure rate: {selected['failure_rate']:.6f}",
        f"- Fake recall after crop: {selected['fake_recall_after_crop']:.6f}",
        f"- Gate result: {gate}",
        "",
        "## G. Phase 2.1 Verdict",
        "",
        verdict,
        "",
        "## H. Exact Next Action",
        "",
        exact_next_action(verdict),
    ]
    return "\n".join(lines) + "\n"


def ranking_table(ranking: list[dict[str, Any]]) -> str:
    lines = [
        "| Rank | Strategy | F->R flips | Class changes | Fake recall | Median diff | Failure rate | Avg ms |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranking:
        lines.append(
            "| {rank} | `{strategy}` | {f2r} | {changes} | {fake_recall:.6f} | {median} | {failure:.6f} | {avg_ms} |".format(
                rank=row["rank"],
                strategy=row["strategy_name"],
                f2r=row["fake_to_real_flips"],
                changes=row["total_class_changes"],
                fake_recall=row["fake_recall_after_crop"],
                median=_fmt(row["median_abs_score_difference"]),
                failure=row["failure_rate"],
                avg_ms=_fmt(row["average_processing_time_ms"]),
            )
        )
    return "\n".join(lines)


def failure_summary_text(failure_payload: dict[str, Any]) -> str:
    examples = failure_payload["saved_examples"]
    if not examples:
        return "No fake-to-real flips occurred in the top three ranked strategies, so no failure-example folders were required."
    cause_counts: dict[str, int] = {}
    for example in examples:
        for cause in example["causes"]:
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
    cause_text = ", ".join(f"{key}: {value}" for key, value in sorted(cause_counts.items()))
    return (
        f"Saved {len(examples)} fake-to-real failure example folders from the top three ranked strategies. "
        f"Geometry-assisted cause tags: {cause_text}. These tags are conservative and are not treated as proof "
        "that the detector caused a failure unless the saved visuals support it."
    )


def choose_selected_candidate(expanded_ranking: list[dict[str, Any]]) -> dict[str, Any]:
    best = expanded_ranking[0]
    for row in expanded_ranking:
        if row["strategy_family"] != "prepared_face_conditional":
            continue
        stability_equivalent = (
            row["fake_to_real_flips"] == best["fake_to_real_flips"]
            and row["total_class_changes"] == best["total_class_changes"]
            and row["failure_rate"] == best["failure_rate"]
            and row["fake_recall_after_crop"] >= best["fake_recall_after_crop"]
            and row["median_abs_score_difference"] == best["median_abs_score_difference"]
            and row["p90_abs_score_difference"] == best["p90_abs_score_difference"]
        )
        if stability_equivalent:
            return row
    return best


def detector_verdict(fifty_payload: dict[str, Any], failure_payload: dict[str, Any]) -> str:
    rows = fifty_payload["rows"]
    top_names = set(failure_payload["top_strategy_names"])
    top_rows = [row for row in rows if row["strategy_name"] in top_names]
    detection_failures = sum(1 for row in top_rows if row.get("status") == "no_face_detected")
    if detection_failures:
        return (
            f"Haar had {detection_failures} detection failures among top-ranked strategies in the 50-image sweep. "
            "A YuNet comparison is justified before API integration."
        )
    examples = failure_payload["saved_examples"]
    detector_like = sum(1 for example in examples if "face_or_crop_near_image_boundary" in example["causes"])
    if examples and detector_like >= max(2, len(examples) // 2):
        return (
            "Haar detected faces in the sweep, but many saved failures show boundary-sensitive geometry. "
            "A controlled YuNet comparison is justified."
        )
    return (
        "Haar was sufficient for this prepared-face crop-sweep sample. The dominant issue is crop/model sensitivity, "
        "not clear detector-box failure. YuNet should not be added yet."
    )


def phase_verdict(selected: dict[str, Any], gate: str, detector_assessment: str) -> str:
    if "YuNet" in detector_assessment and "not be added" not in detector_assessment:
        return "requires YuNet detector comparison"
    if gate == "pass":
        return "ready for API prototype with documented limitations"
    if selected["fake_to_real_flip_rate"] > 0.20:
        return "requires model retraining with crop augmentations"
    return "ready for API prototype with documented limitations"


def exact_next_action(verdict: str) -> str:
    if verdict == "requires YuNet detector comparison":
        return "Run a controlled YuNet-vs-Haar comparison using the selected crop strategy and the same deterministic validation images."
    if verdict == "requires model retraining with crop augmentations":
        return "Start a retraining experiment that includes detector-style crop augmentations while preserving the existing model."
    return "Integrate the selected crop candidate into a local API prototype with the Phase 2.1 limitations documented."


def integration_gate(selected: dict[str, Any]) -> str:
    passes = (
        selected["fake_to_real_flip_rate"] < 0.10
        and selected["total_class_change_rate"] < 0.10
        and selected["failure_rate"] < 0.05
    )
    return "pass" if passes else "fail"


def selected_strategy_text(top_two: list[dict[str, Any]], selected: dict[str, Any]) -> str:
    definitions = {definition["name"]: definition for definition in top_two}
    definition = definitions.get(selected["strategy_name"], {})
    return "```json\n" + json.dumps(definition, indent=2) + "\n```"


def square_crop_box(detection: dict[str, Any], image_width: int, image_height: int, margin: float) -> dict[str, int]:
    face_w = detection["x2"] - detection["x1"]
    face_h = detection["y2"] - detection["y1"]
    side = int(round(max(face_w, face_h) * (1.0 + 2.0 * margin)))
    side = max(1, min(side, image_width, image_height))
    cx = (detection["x1"] + detection["x2"]) / 2.0
    cy = (detection["y1"] + detection["y2"]) / 2.0
    x1 = shift_into_bounds(int(round(cx - side / 2.0)), side, image_width)
    y1 = shift_into_bounds(int(round(cy - side / 2.0)), side, image_height)
    return {"x1": x1, "y1": y1, "x2": x1 + side, "y2": y1 + side}


def expand_detection_box(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
    expansion: dict[str, float],
) -> dict[str, int]:
    face_w = detection["x2"] - detection["x1"]
    face_h = detection["y2"] - detection["y1"]
    x1 = int(round(detection["x1"] - face_w * expansion.get("left", 0.0)))
    y1 = int(round(detection["y1"] - face_h * expansion.get("top", 0.0)))
    x2 = int(round(detection["x2"] + face_w * expansion.get("right", 0.0)))
    y2 = int(round(detection["y2"] + face_h * expansion.get("bottom", 0.0)))
    return clamp_box({"x1": x1, "y1": y1, "x2": x2, "y2": y2}, image_width, image_height)


def square_from_rect(rect: dict[str, int], image_width: int, image_height: int) -> dict[str, int]:
    width = rect["x2"] - rect["x1"]
    height = rect["y2"] - rect["y1"]
    side = max(width, height, 1)
    side = min(side, image_width, image_height)
    cx = (rect["x1"] + rect["x2"]) / 2.0
    cy = (rect["y1"] + rect["y2"]) / 2.0
    x1 = shift_into_bounds(int(round(cx - side / 2.0)), side, image_width)
    y1 = shift_into_bounds(int(round(cy - side / 2.0)), side, image_height)
    return {"x1": x1, "y1": y1, "x2": x1 + side, "y2": y1 + side}


def clamp_box(box: dict[str, int], image_width: int, image_height: int) -> dict[str, int]:
    x1 = max(0, min(image_width, int(box["x1"])))
    y1 = max(0, min(image_height, int(box["y1"])))
    x2 = max(0, min(image_width, int(box["x2"])))
    y2 = max(0, min(image_height, int(box["y2"])))
    if x2 <= x1:
        x2 = min(image_width, x1 + 1)
    if y2 <= y1:
        y2 = min(image_height, y1 + 1)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def crop_image(image: Image.Image, crop_box: dict[str, int]) -> Image.Image:
    return image.crop((crop_box["x1"], crop_box["y1"], crop_box["x2"], crop_box["y2"]))


def make_original_input(image: Image.Image) -> Image.Image:
    return resize_nearest(image)


def resize_nearest(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize(MODEL_INPUT_SIZE, resample=Image.Resampling.NEAREST)


def resize_with_padding(image: Image.Image, padding_method: str) -> tuple[Image.Image, dict[str, int]]:
    image = image.convert("RGB")
    scale = min(MODEL_INPUT_SIZE[0] / image.width, MODEL_INPUT_SIZE[1] / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = image.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
    left = (MODEL_INPUT_SIZE[0] - new_w) // 2
    top = (MODEL_INPUT_SIZE[1] - new_h) // 2
    right = MODEL_INPUT_SIZE[0] - new_w - left
    bottom = MODEL_INPUT_SIZE[1] - new_h - top

    if padding_method == "black":
        canvas = Image.new("RGB", MODEL_INPUT_SIZE, (0, 0, 0))
        canvas.paste(resized, (left, top))
        return canvas, {"left": left, "right": right, "top": top, "bottom": bottom}

    array = np.asarray(resized)
    mode = "edge" if padding_method == "edge" or min(new_w, new_h) < 2 else "reflect"
    padded = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode=mode)
    return Image.fromarray(padded.astype(np.uint8), mode="RGB"), {"left": left, "right": right, "top": top, "bottom": bottom}


def predict_scores(classifier: DeepfakeClassifier, images: list[Image.Image], batch_size: int = 32) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        batch = np.concatenate([preprocess_for_model(image) for image in batch_images], axis=0)
        predictions = classifier.model.predict(batch, verbose=0, batch_size=len(batch_images))
        values = np.asarray(predictions, dtype=np.float32).reshape(-1)
        if len(values) != len(batch_images):
            raise RuntimeError(f"Expected {len(batch_images)} predictions, got {len(values)}")
        for value in values:
            score = float(value)
            if not np.isfinite(score) or score < 0.0 or score > 1.0:
                raise ValueError(f"Unexpected model score: {score}")
            scores.append(score)
    return scores


def face_geometry(image_size: tuple[int, int], detection: dict[str, Any]) -> dict[str, float]:
    width, height = image_size
    face_w = detection["x2"] - detection["x1"]
    face_h = detection["y2"] - detection["y1"]
    face_cx = (detection["x1"] + detection["x2"]) / 2.0
    face_cy = (detection["y1"] + detection["y2"]) / 2.0
    offset_x = abs(face_cx - width / 2.0) / max(width, 1)
    offset_y = abs(face_cy - height / 2.0) / max(height, 1)
    return {
        "face_occupancy_ratio": round((face_w * face_h) / max(width * height, 1), 6),
        "face_center_offset_x": round(offset_x, 6),
        "face_center_offset_y": round(offset_y, 6),
        "face_center_offset_max": round(max(offset_x, offset_y), 6),
    }


def empty_geometry() -> dict[str, Any]:
    return {
        "face_occupancy_ratio": "",
        "face_center_offset_x": "",
        "face_center_offset_y": "",
        "face_center_offset_max": "",
    }


def draw_detected_box(image: Image.Image, detection: dict[str, Any] | None) -> Image.Image:
    annotated = image.copy()
    if detection:
        draw = ImageDraw.Draw(annotated)
        draw.rectangle((detection["x1"], detection["y1"], detection["x2"], detection["y2"]), outline=(255, 0, 0), width=3)
    return annotated


def public_box(detection: dict[str, Any] | None) -> dict[str, int] | None:
    if detection is None:
        return None
    return {key: int(detection[key]) for key in ["x1", "y1", "x2", "y2"]}


def class_from_score(score: float) -> str:
    return "real" if score >= 0.5 else "fake"


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def strategy_artifact_penalty(strategy_name: str, strategy_family: str) -> int:
    if strategy_family == "context_stretch":
        return 3
    if "black" in strategy_name:
        return 3
    if "reflect" in strategy_name:
        return 2
    if "edge" in strategy_name:
        return 1
    return 0


def shift_into_bounds(start: int, side: int, limit: int) -> int:
    if start < 0:
        return 0
    if start + side > limit:
        return limit - side
    return start


def _close_to_square(width: int, height: int, tolerance: float) -> bool:
    if min(width, height) <= 0:
        return False
    return abs(width - height) / max(width, height) <= tolerance


def sanitize_name(relative_path: str) -> str:
    return relative_path.replace("\\", "_").replace("/", "_").replace(":", "_")


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def strategy_to_json(strategy: CropStrategy) -> dict[str, Any]:
    result = {
        "name": strategy.name,
        "family": strategy.family,
        "description": strategy.description,
        "margin": strategy.margin,
        "expansion": strategy.expansion,
        "padding_method": strategy.padding_method,
    }
    if strategy.fallback:
        result["fallback"] = strategy_to_json(strategy.fallback)
    else:
        result["fallback"] = None
    return result


def strategy_from_json(definition: dict[str, Any], available: dict[str, CropStrategy]) -> CropStrategy:
    if definition["name"] in available:
        return available[definition["name"]]
    fallback = strategy_from_json(definition["fallback"], available) if definition.get("fallback") else None
    return CropStrategy(
        name=definition["name"],
        family=definition["family"],
        description=definition["description"],
        margin=definition.get("margin"),
        expansion=definition.get("expansion"),
        padding_method=definition.get("padding_method"),
        fallback=fallback,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if key != "rows"}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2.1 crop strategy sweep.")
    parser.add_argument("--run-all", action="store_true", help="Run 50-image sweep, expanded validation, examples, and report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run_all:
        raise SystemExit("Use --run-all to execute the full deterministic crop sweep.")
    payload = run_all()
    print(json.dumps(_jsonable(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
