"""Evaluate the existing human-face deepfake detector.

This script is intentionally read-only with respect to the saved model and
dataset. It evaluates the current test split and writes reproducible audit
artifacts under reports/existing_model/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CLASS_NAMES = ["fake", "real"]
CLASS_TO_LABEL = {"fake": 0, "real": 1}
SEED = 69


@dataclass
class ModelInspection:
    model_path: str
    model_file_size_bytes: int
    tensorflow_version: str
    keras_version: str
    load_compile_requested: bool
    loaded_with_compile: bool
    load_warning: str | None
    input_shape: Any
    output_shape: Any
    total_params: int
    trainable_params: int
    non_trainable_params: int
    output_activation: str | None
    optimizer: str | None
    loss: str | None
    metrics: list[str]
    flatten_to_dense: dict[str, Any] | None
    layers: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate the saved deepfake detector on the existing splits."
    )
    parser.add_argument("--model", type=Path, default=root / "model" / "deepfake_detector_93acc.h5")
    parser.add_argument("--data-root", type=Path, default=root / "data" / "raw" / "real_vs_fake" / "real-vs-fake")
    parser.add_argument("--test-csv", type=Path, default=root / "data" / "raw" / "test.csv")
    parser.add_argument("--valid-csv", type=Path, default=root / "data" / "raw" / "valid.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "existing_model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip-validation-analysis",
        action="store_true",
        help="Evaluate only test metrics and skip validation-score uncertainty analysis.",
    )
    return parser.parse_args()


def ensure_real_hdf5(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("rb") as handle:
        header = handle.read(8)
    if header.startswith(b"version "):
        raise RuntimeError(f"Model file is a Git LFS pointer, not a model: {path}")
    if not header.startswith(b"\x89HDF"):
        raise RuntimeError(f"Model file does not look like HDF5/Keras data: {path}")


def normalize_shape(shape: Any) -> Any:
    if shape is None:
        return None
    if isinstance(shape, tuple):
        return [normalize_shape(item) for item in shape]
    if isinstance(shape, list):
        return [normalize_shape(item) for item in shape]
    try:
        return int(shape)
    except (TypeError, ValueError):
        return str(shape)


def count_params(weights: list[tf.Variable]) -> int:
    return int(sum(np.prod(weight.shape.as_list()) for weight in weights))


def describe_loss(model: tf.keras.Model) -> str | None:
    loss = getattr(model, "loss", None)
    if loss is None:
        return None
    if isinstance(loss, str):
        return loss
    return getattr(loss, "__name__", loss.__class__.__name__)


def describe_metrics(model: tf.keras.Model) -> list[str]:
    metrics = []
    for metric in getattr(model, "metrics", []) or []:
        name = getattr(metric, "name", None) or metric.__class__.__name__
        metrics.append(str(name))
    compiled = getattr(model, "compiled_metrics", None)
    if compiled is not None:
        for metric in getattr(compiled, "_user_metrics", []) or []:
            if isinstance(metric, str):
                metrics.append(metric)
            else:
                metrics.append(getattr(metric, "name", metric.__class__.__name__))
    return sorted(set(metrics))


def flatten_dense_info(model: tf.keras.Model) -> dict[str, Any] | None:
    layers = list(model.layers)
    for idx, layer in enumerate(layers):
        if layer.__class__.__name__ == "Flatten":
            for next_layer in layers[idx + 1 :]:
                if next_layer.__class__.__name__ == "Dense":
                    kernel_shape = next_layer.weights[0].shape.as_list()
                    input_units = int(kernel_shape[0])
                    output_units = int(kernel_shape[1])
                    dense_params = int(np.prod(kernel_shape) + output_units)
                    return {
                        "flatten_layer": layer.name,
                        "dense_layer": next_layer.name,
                        "flatten_output_units": input_units,
                        "dense_units": output_units,
                        "dense_kernel_params": int(np.prod(kernel_shape)),
                        "dense_bias_params": output_units,
                        "dense_total_params": dense_params,
                        "float32_weight_bytes_without_optimizer": dense_params * 4,
                        "approx_adam_weight_plus_slots_bytes": dense_params * 4 * 3,
                    }
    return None


def inspect_model(model_path: Path, output_dir: Path) -> tuple[tf.keras.Model, ModelInspection]:
    ensure_real_hdf5(model_path)
    load_warning = None
    loaded_with_compile = True
    try:
        model = load_model(model_path, compile=True)
    except Exception as exc:
        load_warning = f"compile=True load failed, fell back to compile=False: {exc}"
        loaded_with_compile = False
        model = load_model(model_path, compile=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
        with redirect_stdout(handle):
            model.summary()

    layers = []
    for layer in model.layers:
        activation = getattr(getattr(layer, "activation", None), "__name__", None)
        layers.append(
            {
                "name": layer.name,
                "class_name": layer.__class__.__name__,
                "input_shape": normalize_shape(getattr(layer, "input_shape", None)),
                "output_shape": normalize_shape(getattr(layer, "output_shape", None)),
                "params": int(layer.count_params()),
                "activation": activation,
            }
        )

    output_activation = None
    if model.layers:
        output_activation = getattr(getattr(model.layers[-1], "activation", None), "__name__", None)

    optimizer = getattr(model, "optimizer", None)
    optimizer_name = optimizer.__class__.__name__ if optimizer is not None else None

    inspection = ModelInspection(
        model_path=str(model_path),
        model_file_size_bytes=model_path.stat().st_size,
        tensorflow_version=tf.__version__,
        keras_version=tf.keras.__version__ if hasattr(tf.keras, "__version__") else "tf.keras",
        load_compile_requested=True,
        loaded_with_compile=loaded_with_compile,
        load_warning=load_warning,
        input_shape=normalize_shape(model.input_shape),
        output_shape=normalize_shape(model.output_shape),
        total_params=int(model.count_params()),
        trainable_params=count_params(model.trainable_weights),
        non_trainable_params=count_params(model.non_trainable_weights),
        output_activation=output_activation,
        optimizer=optimizer_name,
        loss=describe_loss(model),
        metrics=describe_metrics(model),
        flatten_to_dense=flatten_dense_info(model),
        layers=layers,
    )
    with (output_dir / "model_inspection.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(inspection), handle, indent=2)
    return model, inspection


def load_split_csv(csv_path: Path, data_root: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"path", "label", "label_str"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df = df.copy()
    df["label_str"] = df["label_str"].astype(str)
    df["label"] = df["label"].astype(int)
    bad_labels = df[~df["label_str"].isin(CLASS_NAMES)]
    if not bad_labels.empty:
        raise ValueError(f"{csv_path} has unexpected label_str values: {bad_labels['label_str'].unique()}")

    mismatches = df[df["label"] != df["label_str"].map(CLASS_TO_LABEL)]
    if not mismatches.empty:
        raise ValueError(f"{csv_path} has label/label_str mismatches, first rows: {mismatches.head().to_dict('records')}")

    full_paths = df["path"].map(lambda value: data_root / value)
    missing_paths = [str(path) for path in full_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"{len(missing_paths)} image paths are missing. First 10: {missing_paths[:10]}")
    return df


def make_generator(df: pd.DataFrame, data_root: Path, batch_size: int, shuffle: bool = False):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=str(data_root),
        x_col="path",
        y_col="label_str",
        classes=CLASS_NAMES,
        target_size=(256, 256),
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
        interpolation="nearest",
        validate_filenames=True,
    )
    if generator.class_indices != CLASS_TO_LABEL:
        raise ValueError(f"Unexpected class_indices: {generator.class_indices}; expected {CLASS_TO_LABEL}")
    if generator.n != len(df):
        raise ValueError(f"Generator accepted {generator.n} rows but CSV contains {len(df)} rows")
    return generator


def predict_split(
    model: tf.keras.Model,
    df: pd.DataFrame,
    data_root: Path,
    batch_size: int,
) -> pd.DataFrame:
    generator = make_generator(df, data_root, batch_size=batch_size, shuffle=False)
    real_scores = model.predict(generator, verbose=1).reshape(-1).astype(float)
    if len(real_scores) != len(df):
        raise RuntimeError(f"Predicted {len(real_scores)} scores for {len(df)} rows")

    result = df[["path", "label", "label_str"]].copy()
    result = result.rename(columns={"label": "true_label", "label_str": "true_label_str"})
    result["real_score"] = real_scores
    result["fake_score"] = 1.0 - result["real_score"]
    result["predicted_label"] = (result["real_score"] >= 0.5).astype(int)
    result["predicted_label_str"] = np.where(result["predicted_label"] == 1, "real", "fake")
    result["correct"] = result["predicted_label"] == result["true_label"]
    return result


def score_quantiles(values: pd.Series) -> dict[str, float | None]:
    if values.empty:
        return {"count": 0, "min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    quantiles = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(values.shape[0]),
        "min": float(values.min()),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "max": float(values.max()),
    }


def compute_metrics(predictions: pd.DataFrame, threshold: float) -> dict[str, Any]:
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_pred = predictions["predicted_label"].to_numpy(dtype=int)
    real_score = predictions["real_score"].to_numpy(dtype=float)
    fake_score = predictions["fake_score"].to_numpy(dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fake_correct, fake_as_real = [int(value) for value in cm[0]]
    real_as_fake, real_correct = [int(value) for value in cm[1]]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    roc_auc_fake = roc_auc_score((y_true == 0).astype(int), fake_score)

    false_positive_rate = real_as_fake / (real_as_fake + real_correct) if (real_as_fake + real_correct) else 0.0
    false_negative_rate = fake_as_real / (fake_correct + fake_as_real) if (fake_correct + fake_as_real) else 0.0

    return {
        "threshold_real_score": threshold,
        "class_mapping": CLASS_TO_LABEL,
        "model_output": "real_score = model output probability-like score for class 'real'",
        "fake_score_formula": "fake_score = 1.0 - real_score",
        "false_positive_definition": "a real face incorrectly labelled fake",
        "false_negative_definition": "a fake face incorrectly labelled real",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision[0]),
        "recall_fake": float(recall[0]),
        "f1_fake": float(f1[0]),
        "support_fake": int(support[0]),
        "precision_real": float(precision[1]),
        "recall_real": float(recall[1]),
        "f1_real": float(f1[1]),
        "support_real": int(support[1]),
        "roc_auc_fake_score": float(roc_auc_fake),
        "confusion_matrix_labels": CLASS_NAMES,
        "confusion_matrix": cm.astype(int).tolist(),
        "fake_correct_predicted_fake": fake_correct,
        "fake_incorrect_predicted_real": fake_as_real,
        "real_incorrect_predicted_fake": real_as_fake,
        "real_correct_predicted_real": real_correct,
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
    }


def save_plots(predictions: pd.DataFrame, output_dir: Path) -> None:
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_pred = predictions["predicted_label"].to_numpy(dtype=int)
    fake_score = predictions["fake_score"].to_numpy(dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Existing Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve((y_true == 0).astype(int), fake_score)
    auc = roc_auc_score((y_true == 0).astype(int), fake_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Fake-score ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False positive rate (real labelled fake)")
    plt.ylabel("True positive rate (fake labelled fake)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    groups = [
        ("correct real", (predictions["true_label"] == 1) & predictions["correct"]),
        ("incorrect real", (predictions["true_label"] == 1) & ~predictions["correct"]),
        ("correct fake", (predictions["true_label"] == 0) & predictions["correct"]),
        ("incorrect fake", (predictions["true_label"] == 0) & ~predictions["correct"]),
    ]
    plt.figure(figsize=(8, 5))
    for label, mask in groups:
        values = predictions.loc[mask, "real_score"]
        if not values.empty:
            plt.hist(values, bins=40, alpha=0.45, label=f"{label} (n={len(values)})")
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1, label="0.5 threshold")
    plt.xlabel("Model output real_score")
    plt.ylabel("Image count")
    plt.title("Prediction Score Histogram by Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_histogram.png", dpi=160)
    plt.close()


def deterministic_sample(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sample_rows = []
    for label_name, label_value in [("real", 1), ("fake", 0)]:
        indices = predictions.index[predictions["true_label"] == label_value].tolist()
        chosen = rng.sample(indices, 5)
        for idx in chosen:
            row = predictions.loc[idx]
            sample_rows.append(
                {
                    "path": row["path"],
                    "true_label": label_name,
                    "raw_model_output_real_score": float(row["real_score"]),
                    "fake_score": float(row["fake_score"]),
                    "predicted_label": row["predicted_label_str"],
                    "correct": bool(row["correct"]),
                }
            )
    return sample_rows


def outcome_distributions(predictions: pd.DataFrame) -> dict[str, Any]:
    definitions = {
        "correctly_classified_real": (predictions["true_label"] == 1) & predictions["correct"],
        "incorrectly_classified_real": (predictions["true_label"] == 1) & ~predictions["correct"],
        "correctly_classified_fake": (predictions["true_label"] == 0) & predictions["correct"],
        "incorrectly_classified_fake": (predictions["true_label"] == 0) & ~predictions["correct"],
    }
    return {name: score_quantiles(predictions.loc[mask, "real_score"]) for name, mask in definitions.items()}


def validation_uncertainty_analysis(validation_predictions: pd.DataFrame) -> dict[str, Any]:
    ranges = [
        (0.45, 0.55),
        (0.40, 0.60),
        (0.35, 0.65),
    ]
    rows = []
    total = len(validation_predictions)
    for low, high in ranges:
        uncertain = validation_predictions["real_score"].between(low, high, inclusive="both")
        certain = ~uncertain
        certain_predictions = validation_predictions.loc[certain]
        rows.append(
            {
                "real_score_range": [low, high],
                "uncertain_count": int(uncertain.sum()),
                "uncertain_fraction": float(uncertain.mean()),
                "certain_count": int(certain.sum()),
                "certain_fraction": float(certain.sum() / total),
                "accuracy_on_certain": (
                    float(accuracy_score(certain_predictions["true_label"], certain_predictions["predicted_label"]))
                    if not certain_predictions.empty
                    else None
                ),
            }
        )
    return {
        "basis": "validation split score distributions, not test-set threshold tuning",
        "candidate_uncertainty_ranges": rows,
        "score_distributions": {
            "real_images_real_score": score_quantiles(validation_predictions.loc[validation_predictions["true_label"] == 1, "real_score"]),
            "fake_images_real_score": score_quantiles(validation_predictions.loc[validation_predictions["true_label"] == 0, "real_score"]),
        },
    }


def write_summary(
    output_dir: Path,
    inspection: ModelInspection,
    metrics: dict[str, Any],
    samples: list[dict[str, Any]],
    distributions: dict[str, Any],
    validation_analysis: dict[str, Any] | None,
) -> None:
    lines = [
        "# Existing Model Evaluation Summary",
        "",
        "## Model Load",
        "",
        f"- Model path: `{inspection.model_path}`",
        f"- File size: {inspection.model_file_size_bytes:,} bytes",
        f"- TensorFlow version: `{inspection.tensorflow_version}`",
        f"- Loaded with compile metadata: `{inspection.loaded_with_compile}`",
        f"- Input shape: `{inspection.input_shape}`",
        f"- Output shape: `{inspection.output_shape}`",
        f"- Output activation: `{inspection.output_activation}`",
        f"- Optimizer: `{inspection.optimizer}`",
        f"- Loss: `{inspection.loss}`",
        f"- Stored metrics: `{inspection.metrics}`",
        "",
        "## Metrics On Test Split",
        "",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.6f}",
        f"- Precision fake: {metrics['precision_fake']:.6f}",
        f"- Recall fake: {metrics['recall_fake']:.6f}",
        f"- F1 fake: {metrics['f1_fake']:.6f}",
        f"- Precision real: {metrics['precision_real']:.6f}",
        f"- Recall real: {metrics['recall_real']:.6f}",
        f"- F1 real: {metrics['f1_real']:.6f}",
        f"- ROC-AUC using fake score: {metrics['roc_auc_fake_score']:.6f}",
        f"- False positive rate: {metrics['false_positive_rate']:.6f}",
        f"- False negative rate: {metrics['false_negative_rate']:.6f}",
        f"- Confusion matrix labels: `{metrics['confusion_matrix_labels']}`",
        f"- Confusion matrix: `{metrics['confusion_matrix']}`",
        "",
        "False positive means a real face was incorrectly labelled fake. False negative means a fake face was incorrectly labelled real.",
        "",
        "## Deterministic Individual Predictions",
        "",
        "| Path | True label | Raw real_score | Fake score | Predicted label | Correct |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for sample in samples:
        lines.append(
            "| {path} | {true_label} | {real:.6f} | {fake:.6f} | {predicted} | {correct} |".format(
                path=sample["path"],
                true_label=sample["true_label"],
                real=sample["raw_model_output_real_score"],
                fake=sample["fake_score"],
                predicted=sample["predicted_label"],
                correct=sample["correct"],
            )
        )
    lines.extend(
        [
            "",
            "## Score Distributions",
            "",
            "```json",
            json.dumps(distributions, indent=2),
            "```",
        ]
    )
    if validation_analysis is not None:
        lines.extend(
            [
                "",
                "## Validation-Based Uncertainty Probe",
                "",
                "This is an analysis probe only; no permanent threshold is adopted here.",
                "",
                "```json",
                json.dumps(validation_analysis, indent=2),
                "```",
            ]
        )
    (output_dir / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, inspection = inspect_model(args.model, args.output_dir)

    test_df = load_split_csv(args.test_csv, args.data_root)
    predictions = predict_split(model, test_df, args.data_root, args.batch_size)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    metrics = compute_metrics(predictions, threshold=args.threshold)
    metrics["model_inspection"] = asdict(inspection)
    metrics["test_rows"] = int(len(test_df))
    metrics["claimed_accuracy_from_filename"] = "93acc"
    metrics["reproduction_status"] = (
        "approximately reproduced"
        if abs(metrics["accuracy"] - 0.93) <= 0.01
        else "higher than expected"
        if metrics["accuracy"] > 0.94
        else "lower than expected"
        if metrics["accuracy"] < 0.92
        else "approximately reproduced"
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    report = classification_report(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=6,
        zero_division=0,
    )
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    save_plots(predictions, args.output_dir)
    samples = deterministic_sample(predictions)
    distributions = outcome_distributions(predictions)
    (args.output_dir / "score_distributions.json").write_text(json.dumps(distributions, indent=2), encoding="utf-8")
    (args.output_dir / "sample_predictions.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    validation_analysis = None
    if not args.skip_validation_analysis:
        valid_df = load_split_csv(args.valid_csv, args.data_root)
        validation_predictions = predict_split(model, valid_df, args.data_root, args.batch_size)
        validation_predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        validation_analysis = validation_uncertainty_analysis(validation_predictions)
        (args.output_dir / "validation_uncertainty_analysis.json").write_text(
            json.dumps(validation_analysis, indent=2), encoding="utf-8"
        )

        plt.figure(figsize=(8, 5))
        for label_value, label_name in [(0, "fake"), (1, "real")]:
            values = validation_predictions.loc[validation_predictions["true_label"] == label_value, "real_score"]
            plt.hist(values, bins=50, alpha=0.5, label=f"{label_name} (n={len(values)})")
        plt.axvline(0.5, color="black", linestyle="--", linewidth=1, label="0.5 threshold")
        plt.xlabel("Model output real_score")
        plt.ylabel("Image count")
        plt.title("Validation Score Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "validation_score_histogram.png", dpi=160)
        plt.close()

    write_summary(args.output_dir, inspection, metrics, samples, distributions, validation_analysis)
    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
