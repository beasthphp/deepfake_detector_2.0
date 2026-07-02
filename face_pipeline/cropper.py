from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from face_pipeline.detector import ImageInput, clamp_box, load_rgb_image


SELECTED_CROP_STRATEGY = "preserve_or_context_full_head_l40_r40_t70_b35_square"
CONTEXT_FULL_HEAD_STRATEGY = "context_full_head_l40_r40_t70_b35_square"
CONTEXT_FULL_HEAD_EXPANSION = {"left": 0.40, "right": 0.40, "top": 0.70, "bottom": 0.35}


@dataclass(frozen=True)
class CropConfig:
    margin: float = 0.20
    minimum_crop_size: int = 64


@dataclass(frozen=True)
class PreserveConfig:
    square_tolerance: float = 0.10
    centered_offset_threshold: float = 0.12
    face_occupancy_threshold: float = 0.30
    boundary_fraction: float = 0.015


@dataclass(frozen=True)
class SelectedCropResult:
    crop: Image.Image
    crop_box: dict[str, int]
    crop_strategy: str
    preserved_original: bool
    preserve_decision: dict[str, Any]
    proposed_expanded_crop_box: dict[str, int] | None
    expanded_clamped_box: dict[str, int] | None


def square_crop_box(
    face_box: dict[str, Any],
    image_width: int,
    image_height: int,
    margin: float = 0.20,
    minimum_crop_size: int = 64,
) -> dict[str, int]:
    """Return a square crop around a face, clamped to image bounds.

    `margin=0.20` means roughly 20 percent of the face size is requested on
    each side before clamping. Coordinates use x1/y1/x2/y2 exclusive corners.
    """

    if margin < 0:
        raise ValueError("margin must be >= 0")
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")

    x1 = int(round(float(face_box["x1"])))
    y1 = int(round(float(face_box["y1"])))
    x2 = int(round(float(face_box["x2"])))
    y2 = int(round(float(face_box["y2"])))

    face_width = max(0, x2 - x1)
    face_height = max(0, y2 - y1)
    if face_width <= 0 or face_height <= 0:
        raise ValueError(f"Invalid face box with zero area: {face_box}")

    requested_side = int(round(max(face_width, face_height) * (1.0 + 2.0 * margin)))
    side = max(minimum_crop_size, requested_side)
    side = min(side, image_width, image_height)
    if side < minimum_crop_size:
        raise ValueError(
            f"Image is too small for minimum crop size {minimum_crop_size}: {image_width}x{image_height}"
        )

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = int(round(center_x - side / 2.0))
    crop_y1 = int(round(center_y - side / 2.0))

    crop_x1 = _shift_into_bounds(crop_x1, side, image_width)
    crop_y1 = _shift_into_bounds(crop_y1, side, image_height)
    crop_x2 = crop_x1 + side
    crop_y2 = crop_y1 + side

    return {"x1": int(crop_x1), "y1": int(crop_y1), "x2": int(crop_x2), "y2": int(crop_y2)}


def crop_face(
    image: ImageInput,
    face_box: dict[str, Any],
    config: CropConfig | None = None,
) -> tuple[Image.Image, dict[str, int]]:
    """Crop a detected face with margin and return `(crop, crop_box)`."""

    crop_config = config or CropConfig()
    pil_image = load_rgb_image(image)
    crop_box = square_crop_box(
        face_box=face_box,
        image_width=pil_image.width,
        image_height=pil_image.height,
        margin=crop_config.margin,
        minimum_crop_size=crop_config.minimum_crop_size,
    )
    width = crop_box["x2"] - crop_box["x1"]
    height = crop_box["y2"] - crop_box["y1"]
    if width < crop_config.minimum_crop_size or height < crop_config.minimum_crop_size:
        raise ValueError(f"Rejected too-small crop: {crop_box}")
    if width != height:
        raise ValueError(f"Expected square crop, got {width}x{height}: {crop_box}")
    return pil_image.crop((crop_box["x1"], crop_box["y1"], crop_box["x2"], crop_box["y2"])), crop_box


def crop_face_selected_strategy(
    image: ImageInput,
    detection: dict[str, Any],
    detections: list[dict[str, Any]] | None = None,
    preserve_config: PreserveConfig | None = None,
) -> SelectedCropResult:
    """Apply the selected Phase 2.1 crop strategy for one detected face.

    The strategy preserves already-prepared face crops. Otherwise it applies
    the selected full-head contextual square crop.
    """

    pil_image = load_rgb_image(image)
    all_detections = detections if detections is not None else [detection]
    preserve_decision = should_preserve_original(
        (pil_image.width, pil_image.height),
        all_detections,
        preserve_config,
    )
    if preserve_decision["preserve_original"]:
        crop_box = {"x1": 0, "y1": 0, "x2": pil_image.width, "y2": pil_image.height}
        return SelectedCropResult(
            crop=pil_image.copy(),
            crop_box=crop_box,
            crop_strategy=SELECTED_CROP_STRATEGY,
            preserved_original=True,
            preserve_decision=preserve_decision,
            proposed_expanded_crop_box=None,
            expanded_clamped_box=None,
        )

    boxes = contextual_full_head_crop_boxes(detection, pil_image.width, pil_image.height)
    crop_box = boxes["final_clamped_crop_box"]
    crop = pil_image.crop((crop_box["x1"], crop_box["y1"], crop_box["x2"], crop_box["y2"]))
    return SelectedCropResult(
        crop=crop,
        crop_box=crop_box,
        crop_strategy=CONTEXT_FULL_HEAD_STRATEGY,
        preserved_original=False,
        preserve_decision=preserve_decision,
        proposed_expanded_crop_box=boxes["proposed_expanded_crop_box"],
        expanded_clamped_box=boxes["expanded_clamped_box"],
    )


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


def contextual_full_head_crop_boxes(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
    expansion: dict[str, float] | None = None,
) -> dict[str, dict[str, int]]:
    """Return raw expanded, clamped expanded, and final square crop boxes."""

    expansion_config = expansion or CONTEXT_FULL_HEAD_EXPANSION
    face_w = int(detection["x2"] - detection["x1"])
    face_h = int(detection["y2"] - detection["y1"])
    proposed = {
        "x1": int(round(detection["x1"] - face_w * expansion_config.get("left", 0.0))),
        "y1": int(round(detection["y1"] - face_h * expansion_config.get("top", 0.0))),
        "x2": int(round(detection["x2"] + face_w * expansion_config.get("right", 0.0))),
        "y2": int(round(detection["y2"] + face_h * expansion_config.get("bottom", 0.0))),
    }
    expanded_clamped = clamp_box(proposed, image_width, image_height)
    final_crop = square_from_rect(expanded_clamped, image_width, image_height)
    return {
        "proposed_expanded_crop_box": proposed,
        "expanded_clamped_box": expanded_clamped,
        "final_clamped_crop_box": final_crop,
    }


def square_from_rect(rect: dict[str, int], image_width: int, image_height: int) -> dict[str, int]:
    width = rect["x2"] - rect["x1"]
    height = rect["y2"] - rect["y1"]
    side = max(width, height, 1)
    side = min(side, image_width, image_height)
    cx = (rect["x1"] + rect["x2"]) / 2.0
    cy = (rect["y1"] + rect["y2"]) / 2.0
    x1 = _shift_into_bounds(int(round(cx - side / 2.0)), side, image_width)
    y1 = _shift_into_bounds(int(round(cy - side / 2.0)), side, image_height)
    return {"x1": x1, "y1": y1, "x2": x1 + side, "y2": y1 + side}


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


def _shift_into_bounds(start: int, side: int, limit: int) -> int:
    if start < 0:
        return 0
    if start + side > limit:
        return limit - side
    return start


def _close_to_square(width: int, height: int, tolerance: float) -> bool:
    if min(width, height) <= 0:
        return False
    return abs(width - height) / max(width, height) <= tolerance
