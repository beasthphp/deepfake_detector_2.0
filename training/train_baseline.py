from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from training.config import (
    DEFAULT_TRAINING,
    EXPERIMENT_MODEL_DIR,
    REPORT_DIR,
    RETRAINED_BEST_MODEL,
    SEED,
    SMOKE_BEST_MODEL,
    TRAIN_CSV,
    VALID_CSV,
)
from training.data_pipeline import (
    balanced_subset,
    load_split_csv,
    make_dataset,
    validate_image_files,
    verify_expected_counts,
    verify_paths_exist,
)
from training.models import build_custom_cnn, compile_custom_cnn
from training.utils import dataframe_counts, ensure_directories, model_summary_text, set_global_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or smoke-test the custom CNN baseline.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true", help="Run the small balanced one-epoch smoke test.")
    mode.add_argument("--full", action="store_true", help="Run full training on train.csv with valid.csv validation.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING.batch_size)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAINING.learning_rate)
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_TRAINING.early_stopping_patience)
    parser.add_argument("--reduce-lr-patience", type=int, default=DEFAULT_TRAINING.reduce_lr_patience)
    parser.add_argument("--smoke-train-per-class", type=int, default=DEFAULT_TRAINING.smoke_train_per_class)
    parser.add_argument("--smoke-valid-per-class", type=int, default=DEFAULT_TRAINING.smoke_valid_per_class)
    parser.add_argument("--max-failed-images", type=int, default=DEFAULT_TRAINING.max_failed_images)
    parser.add_argument("--skip-image-validation", action="store_true", help="Skip PIL verify preflight.")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training augmentations.")
    return parser.parse_args()


def build_callbacks(checkpoint_path: Path, history_csv: Path, args: argparse.Namespace) -> list[tf.keras.callbacks.Callback]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=DEFAULT_TRAINING.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(filename=str(history_csv), append=False),
    ]


def inspect_first_batch(dataset: tf.data.Dataset) -> dict[str, object]:
    images, labels = next(iter(dataset))
    labels_np = labels.numpy().astype(int)
    unique, counts = np.unique(labels_np, return_counts=True)
    return {
        "image_shape": [int(value) for value in images.shape],
        "label_shape": [int(value) for value in labels.shape],
        "label_distribution": {str(int(label)): int(count) for label, count in zip(unique, counts)},
        "image_dtype": str(images.dtype.name),
        "image_min": float(tf.reduce_min(images).numpy()),
        "image_max": float(tf.reduce_max(images).numpy()),
    }


def main() -> int:
    args = parse_args()
    set_global_seed(SEED)
    ensure_directories([EXPERIMENT_MODEL_DIR, REPORT_DIR])

    train_df_full = load_split_csv(TRAIN_CSV, "train")
    valid_df_full = load_split_csv(VALID_CSV, "valid")
    train_counts = verify_expected_counts(train_df_full, "train")
    valid_counts = verify_expected_counts(valid_df_full, "valid")
    verify_paths_exist(train_df_full)
    verify_paths_exist(valid_df_full)

    if args.smoke_test:
        mode_name = "smoke"
        train_df = balanced_subset(train_df_full, args.smoke_train_per_class)
        valid_df = balanced_subset(valid_df_full, args.smoke_valid_per_class)
        epochs = 1
        checkpoint_path = SMOKE_BEST_MODEL
        history_csv = REPORT_DIR / "training_history_smoke.csv"
        summary_path = REPORT_DIR / "smoke_summary.md"
        metadata_path = REPORT_DIR / "smoke_metadata.json"
    else:
        mode_name = "full"
        train_df = train_df_full.reset_index(drop=True)
        valid_df = valid_df_full.reset_index(drop=True)
        epochs = args.epochs or DEFAULT_TRAINING.max_epochs
        checkpoint_path = RETRAINED_BEST_MODEL
        history_csv = REPORT_DIR / "training_history.csv"
        summary_path = REPORT_DIR / "training_summary.md"
        metadata_path = REPORT_DIR / "training_metadata.json"

    if not args.skip_image_validation:
        train_validation = validate_image_files(
            train_df,
            REPORT_DIR / f"unreadable_{mode_name}_train_images.csv",
            max_failed_images=args.max_failed_images,
        )
        valid_validation = validate_image_files(
            valid_df,
            REPORT_DIR / f"unreadable_{mode_name}_valid_images.csv",
            max_failed_images=args.max_failed_images,
        )
    else:
        train_validation = {"checked": 0, "failed": 0, "skipped": True}
        valid_validation = {"checked": 0, "failed": 0, "skipped": True}

    train_dataset = make_dataset(
        train_df,
        batch_size=args.batch_size,
        training=True,
        seed=SEED,
        use_augmentation=not args.no_augmentation,
    )
    valid_dataset = make_dataset(
        valid_df,
        batch_size=args.batch_size,
        training=False,
        seed=SEED,
        use_augmentation=False,
    )

    train_batch = inspect_first_batch(train_dataset)
    valid_batch = inspect_first_batch(valid_dataset)

    model = compile_custom_cnn(build_custom_cnn(), learning_rate=args.learning_rate)
    summary_text = model_summary_text(model)
    model_summary_file = REPORT_DIR / f"{mode_name}_model_summary.txt"
    model_summary_file.write_text(summary_text, encoding="utf-8")

    callbacks = build_callbacks(checkpoint_path, history_csv, args)

    start = time.time()
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed_seconds = time.time() - start

    history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
    val_auc_values = history_dict.get("val_auc", [])
    best_validation_epoch = int(np.argmax(val_auc_values) + 1) if val_auc_values else None
    best_validation_auc = float(max(val_auc_values)) if val_auc_values else None

    metadata = {
        "mode": mode_name,
        "seed": SEED,
        "batch_size": args.batch_size,
        "epochs_requested": epochs,
        "epochs_completed": len(history.epoch),
        "learning_rate": args.learning_rate,
        "augmentation_enabled": not args.no_augmentation,
        "full_split_counts": {"train": train_counts, "valid": valid_counts},
        "used_split_counts": {"train": dataframe_counts(train_df), "valid": dataframe_counts(valid_df)},
        "train_batch": train_batch,
        "valid_batch": valid_batch,
        "train_image_validation": train_validation,
        "valid_image_validation": valid_validation,
        "checkpoint_path": str(checkpoint_path),
        "history_csv": str(history_csv),
        "model_summary_file": str(model_summary_file),
        "best_validation_epoch": best_validation_epoch,
        "best_validation_auc": best_validation_auc,
        "elapsed_seconds": elapsed_seconds,
        "history": history_dict,
    }
    write_json(metadata_path, metadata)

    lines = [
        f"# {mode_name.title()} Baseline Training Summary",
        "",
        f"- Mode: `{mode_name}`",
        f"- Seed: {SEED}",
        f"- Batch size: {args.batch_size}",
        f"- Epochs completed: {len(history.epoch)}",
        f"- Augmentation enabled: `{not args.no_augmentation}`",
        f"- Full train counts: `{train_counts}`",
        f"- Full validation counts: `{valid_counts}`",
        f"- Used train counts: `{metadata['used_split_counts']['train']}`",
        f"- Used validation counts: `{metadata['used_split_counts']['valid']}`",
        f"- Train batch: `{train_batch}`",
        f"- Validation batch: `{valid_batch}`",
        f"- Checkpoint path: `{checkpoint_path}`",
        f"- Best validation epoch: `{best_validation_epoch}`",
        f"- Best validation ROC-AUC: `{best_validation_auc}`",
        f"- Elapsed seconds: `{elapsed_seconds:.2f}`",
        "",
        "## Final Epoch Metrics",
        "",
    ]
    for key, values in history_dict.items():
        if values:
            lines.append(f"- {key}: {values[-1]:.6f}")
    lines.extend(["", "## Model Summary", "", "```text", summary_text.rstrip(), "```"])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
