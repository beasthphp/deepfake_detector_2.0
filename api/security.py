from __future__ import annotations

import io
import warnings
from typing import Any

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from api.config import APIConfig
from api.exceptions import ImageValidationError


SUPPORTED_CONTENT_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


async def read_and_decode_upload(file: UploadFile, config: APIConfig) -> tuple[Image.Image, dict[str, Any], float]:
    import time

    started = time.perf_counter()
    declared_type = (file.content_type or "").split(";")[0].strip().lower()
    if declared_type not in SUPPORTED_CONTENT_TYPES:
        raise ImageValidationError(415, "unsupported_content_type", "Unsupported image content type.")

    content = await file.read(config.max_upload_bytes + 1)
    if not content:
        raise ImageValidationError(400, "empty_upload", "Uploaded file is empty.")
    if len(content) > config.max_upload_bytes:
        raise ImageValidationError(413, "upload_too_large", "Uploaded image exceeds the configured size limit.")

    image = decode_image_bytes(content, config)
    info = {
        "width": image.width,
        "height": image.height,
        "format": image.info.get("source_format", "UNKNOWN"),
    }
    return image, info, round((time.perf_counter() - started) * 1000.0, 3)


def decode_image_bytes(content: bytes, config: APIConfig) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = config.max_image_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as loaded:
                image_format = loaded.format or "UNKNOWN"
                if image_format not in SUPPORTED_FORMATS:
                    raise ImageValidationError(415, "unsupported_image_format", "Unsupported decoded image format.")
                width, height = loaded.size
                validate_dimensions(width, height, config)
                loaded.load()
                rgb = loaded.convert("RGB")
                rgb.info.clear()
                rgb.info["source_format"] = image_format
                return rgb
    except ImageValidationError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(413, "image_too_large", "Decoded image exceeds the pixel limit.") from exc
    except (Image.DecompressionBombWarning, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(400, "malformed_image", "Uploaded file could not be decoded as a supported image.") from exc


def validate_dimensions(width: int, height: int, config: APIConfig) -> None:
    if width < config.min_image_width or height < config.min_image_height:
        raise ImageValidationError(422, "image_too_small", "Image dimensions are below the minimum processing size.")
    if width > config.max_image_width or height > config.max_image_height:
        raise ImageValidationError(413, "image_dimensions_too_large", "Image dimensions exceed the configured limit.")
    if width * height > config.max_image_pixels:
        raise ImageValidationError(413, "image_pixel_count_too_large", "Image pixel count exceeds the configured limit.")
