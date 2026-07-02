from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from training.config import CLASS_NAMES, CLASS_TO_LABEL, EXPECTED_COUNTS, IMAGE_ROOT, IMAGE_SIZE, SEED


AUTOTUNE = tf.data.AUTOTUNE
RESIZE_METHOD = tf.image.ResizeMethod.NEAREST_NEIGHBOR


def load_split_csv(csv_path: Path, split_name: str, image_root: Path = IMAGE_ROOT) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"path", "label", "label_str"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df = df.copy()
    df["label_str"] = df["label_str"].astype(str)
    df["label"] = df["label"].astype(int)
    df["full_path"] = df["path"].map(lambda value: str((image_root / value).resolve()))
    df["split"] = split_name

    unexpected = sorted(set(df["label_str"]) - set(CLASS_NAMES))
    if unexpected:
        raise ValueError(f"{csv_path} contains unexpected label_str values: {unexpected}")

    expected_labels = df["label_str"].map(CLASS_TO_LABEL)
    mismatches = df[df["label"] != expected_labels]
    if not mismatches.empty:
        preview = mismatches[["path", "label", "label_str"]].head(10).to_dict("records")
        raise ValueError(f"{csv_path} has label/label_str mismatches: {preview}")
    return df


def verify_expected_counts(df: pd.DataFrame, split_name: str) -> dict[str, int]:
    counts = {name: int((df["label_str"] == name).sum()) for name in CLASS_NAMES}
    expected = EXPECTED_COUNTS[split_name]
    if counts != expected:
        raise ValueError(f"{split_name} class counts are {counts}, expected {expected}")
    return counts


def verify_paths_exist(df: pd.DataFrame) -> None:
    missing = [path for path in df["full_path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image paths are missing. First 10: {missing[:10]}")


def balanced_subset(df: pd.DataFrame, per_class: int, seed: int = SEED) -> pd.DataFrame:
    parts = []
    for class_name in CLASS_NAMES:
        class_df = df[df["label_str"] == class_name]
        if len(class_df) < per_class:
            raise ValueError(f"Need {per_class} {class_name} rows but only found {len(class_df)}")
        parts.append(class_df.sample(n=per_class, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def validate_image_files(
    df: pd.DataFrame,
    log_path: Path,
    max_failed_images: int = 0,
) -> dict[str, int]:
    failures: list[dict[str, str]] = []
    for record in df[["path", "full_path"]].to_dict("records"):
        full_path = Path(record["full_path"])
        try:
            with Image.open(full_path) as image:
                image.verify()
        except Exception as exc:  # noqa: BLE001 - this is a preflight validator that reports exact files.
            failures.append({"path": record["path"], "full_path": str(full_path), "error": repr(exc)})

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if failures:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "full_path", "error"])
            writer.writeheader()
            writer.writerows(failures)
    else:
        log_path.write_text("path,full_path,error\n", encoding="utf-8")

    if len(failures) > max_failed_images:
        raise RuntimeError(f"{len(failures)} unreadable images found; max allowed is {max_failed_images}. See {log_path}")
    return {"checked": int(len(df)), "failed": len(failures)}


def preprocess_path(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE, method=RESIZE_METHOD)
    image = tf.ensure_shape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    label = tf.cast(label, tf.float32)
    return image, label


def _maybe_apply(probability: float, image: tf.Tensor, fn) -> tf.Tensor:
    return tf.cond(tf.random.uniform(()) < probability, lambda: fn(image), lambda: image)


def _jpeg_compression(image: tf.Tensor) -> tf.Tensor:
    image_uint8 = tf.image.convert_image_dtype(tf.clip_by_value(image, 0.0, 1.0), tf.uint8)
    compressed = tf.image.random_jpeg_quality(image_uint8, min_jpeg_quality=82, max_jpeg_quality=95)
    decoded = tf.image.convert_image_dtype(compressed, tf.float32)
    return tf.ensure_shape(decoded, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])


def _mild_blur(image: tf.Tensor) -> tf.Tensor:
    batched = tf.expand_dims(image, axis=0)
    blurred = tf.nn.avg_pool2d(batched, ksize=3, strides=1, padding="SAME")
    return tf.squeeze(blurred, axis=0)


def _resize_degradation(image: tf.Tensor) -> tf.Tensor:
    scale = tf.random.uniform((), minval=0.82, maxval=0.95)
    small_h = tf.cast(tf.round(tf.cast(IMAGE_SIZE[0], tf.float32) * scale), tf.int32)
    small_w = tf.cast(tf.round(tf.cast(IMAGE_SIZE[1], tf.float32) * scale), tf.int32)
    degraded = tf.image.resize(image, [small_h, small_w], method=tf.image.ResizeMethod.AREA)
    degraded = tf.image.resize(degraded, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return tf.ensure_shape(degraded, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])


def augment_image(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.06)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = _maybe_apply(0.30, image, _jpeg_compression)
    image = _maybe_apply(0.20, image, _mild_blur)
    image = _maybe_apply(0.20, image, _resize_degradation)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_augmentation_layers(seed: int = SEED) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(factor=0.025, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomZoom(height_factor=(-0.04, 0.04), width_factor=(-0.04, 0.04), fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03, fill_mode="reflect", seed=seed),
        ],
        name="training_augmentation",
    )


def make_dataset(
    df: pd.DataFrame,
    batch_size: int,
    training: bool,
    seed: int = SEED,
    use_augmentation: bool = True,
) -> tf.data.Dataset:
    paths = df["full_path"].astype(str).to_numpy()
    labels = df["label"].astype(np.float32).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=min(len(df), 4096), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(preprocess_path, num_parallel_calls=AUTOTUNE)
    if training and use_augmentation:
        augmentation_layers = build_augmentation_layers(seed=seed)

        def apply_training_aug(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            image = augmentation_layers(image, training=True)
            return augment_image(image, label)

        dataset = dataset.map(apply_training_aug, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def label_distribution(labels: Iterable[float]) -> dict[str, int]:
    values, counts = np.unique(np.asarray(list(labels), dtype=int), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}
