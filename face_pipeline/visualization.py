from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from face_pipeline.detector import ImageInput, load_rgb_image


def draw_annotations(
    image: ImageInput,
    result: dict[str, Any],
    output_path: str | Path | None = None,
    show_scores: bool = True,
) -> Image.Image:
    """Draw face boxes and labels on a copy of the original image."""

    annotated = load_rgb_image(image).copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for face in result.get("faces", []):
        box = face.get("bounding_box") or face
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        face_index = face.get("face_index", 0)
        label = face.get("label", "Face")
        if show_scores and "real_score" in face:
            text = f"{face_index}: {label} {face['real_score']:.2f}"
        else:
            text = f"{face_index}: {label}"

        color = _color_for_label(label)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_x = max(0, min(x1, annotated.width - text_width - 4))
        label_y = y1 - text_height - 6
        if label_y < 0:
            label_y = min(y1 + 4, annotated.height - text_height - 4)
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 4, label_y + text_height + 4),
            fill=color,
        )
        draw.text((label_x + 2, label_y + 2), text, fill=(255, 255, 255), font=font)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(path)

    return annotated


def _color_for_label(label: str) -> tuple[int, int, int]:
    if label == "Likely Fake":
        return (210, 55, 55)
    if label == "Likely Real":
        return (40, 145, 95)
    if label == "Uncertain":
        return (190, 130, 35)
    return (45, 105, 210)
