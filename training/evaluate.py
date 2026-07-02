from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

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

from training.config import (
    CLASS_NAMES,
    CLASS_TO_LABEL,
    COMPARISON_REPORT,
    EXISTING_MODEL_METRICS,
    REPORT_DIR,
    RETRAINED_BEST_MODEL,
    SEED,
    TEST_CSV,
)
from training.data_pipeline import load_split_csv, make_dataset, validate_image_files, verify_expected_counts, verify_paths_exist
from training.utils import plot_training_curves, read_json, set_global_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a retrained custom CNN checkpoint.")
    parser.add_argument("--model", type=Path, default=RETRAINED_BEST_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--history-csv", type=Path, default=REPORT_DIR / "training_history.csv")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--skip-image-validation", action="store_true")
    parser.add_argument("--max-failed-images", type=int, default=0)
    return parser.parse_args()


def compute_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_pred = predictions["predicted_label"].to_numpy(dtype=int)
    fake_score = predictions["fake_score"].to_numpy(dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fake_correct, fake_as_real = [int(value) for value in cm[0]]
    real_as_fake, real_correct = [int(value) for value in cm[1]]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    false_positive_rate = real_as_fake / (real_as_fake + real_correct) if (real_as_fake + real_correct) else 0.0
    false_negative_rate = fake_as_real / (fake_correct + fake_as_real) if (fake_correct + fake_as_real) else 0.0
    return {
        "class_mapping": CLASS_TO_LABEL,
        "model_output": "real_score = model output probability-like score for class 'real'",
        "fake_score_formula": "fake_score = 1.0 - real_score",
        "false_positive_definition": "a real image incorrectly predicted fake",
        "false_negative_definition": "a fake image incorrectly predicted real",
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
        "roc_auc_fake_score": float(roc_auc_score((y_true == 0).astype(int), fake_score)),
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
    plt.title("Retrained Baseline Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve((y_true == 0).astype(int), fake_score)
    auc = roc_auc_score((y_true == 0).astype(int), fake_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Fake-score ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False positive rate (real predicted fake)")
    plt.ylabel("True positive rate (fake predicted fake)")
    plt.title("Retrained Baseline ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()


def write_summary(output_dir: Path, model_path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Retrained Baseline Evaluation Summary",
        "",
        f"- Model path: `{model_path}`",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.6f}",
        f"- ROC-AUC using fake score: {metrics['roc_auc_fake_score']:.6f}",
        f"- Precision fake: {metrics['precision_fake']:.6f}",
        f"- Recall fake: {metrics['recall_fake']:.6f}",
        f"- Precision real: {metrics['precision_real']:.6f}",
        f"- Recall real: {metrics['recall_real']:.6f}",
        f"- False positive rate: {metrics['false_positive_rate']:.6f}",
        f"- False negative rate: {metrics['false_negative_rate']:.6f}",
        f"- Confusion matrix labels: `{metrics['confusion_matrix_labels']}`",
        f"- Confusion matrix: `{metrics['confusion_matrix']}`",
        "",
        "False positive means a real image was incorrectly predicted fake. False negative means a fake image was incorrectly predicted real.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(model_path: Path, metrics: dict[str, Any]) -> None:
    existing = read_json(EXISTING_MODEL_METRICS) if EXISTING_MODEL_METRICS.exists() else None
    model_size = model_path.stat().st_size if model_path.exists() else None
    lines = [
        "# Baseline Comparison",
        "",
        "Scope: prepared 140K Real and Fake Faces test split only. This does not prove real-world deepfake performance.",
        "",
        "| Metric | Original CNN | Retrained custom CNN |",
        "| --- | ---: | ---: |",
    ]
    if existing:
        rows = [
            ("Test accuracy", existing["accuracy"], metrics["accuracy"]),
            ("ROC-AUC", existing["roc_auc_fake_score"], metrics["roc_auc_fake_score"]),
            ("Fake precision", existing["precision_fake"], metrics["precision_fake"]),
            ("Fake recall", existing["recall_fake"], metrics["recall_fake"]),
            ("Real precision", existing["precision_real"], metrics["precision_real"]),
            ("Real recall", existing["recall_real"], metrics["recall_real"]),
            ("False-positive rate", existing["false_positive_rate"], metrics["false_positive_rate"]),
            ("False-negative rate", existing["false_negative_rate"], metrics["false_negative_rate"]),
        ]
        for name, original, retrained in rows:
            lines.append(f"| {name} | {original:.6f} | {retrained:.6f} |")
        original_params = existing.get("model_inspection", {}).get("total_params", "unknown")
        original_size = existing.get("model_inspection", {}).get("model_file_size_bytes", "unknown")
    else:
        original_params = "unknown"
        original_size = "unknown"
        lines.append("| Test accuracy | unknown | {:.6f} |".format(metrics["accuracy"]))

    lines.extend(
        [
            f"| Parameter count | {original_params} | {metrics.get('parameter_count', 'unknown')} |",
            f"| Model file size bytes | {original_size} | {model_size} |",
            "",
            "## Notes",
            "",
            "- The original model used the same custom CNN architecture but was saved as a large HDF5 artifact with optimizer state.",
            "- The retrained model uses explicit train/validation/test CSVs and does not use `validation_split`.",
            "- Training stability and overfitting should be interpreted from `reports/retrained_baseline/training_history.csv` and the generated curves.",
            "- Neither result should be claimed to generalize to unrelated real-world deepfake sources.",
        ]
    )
    COMPARISON_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_global_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    test_df = load_split_csv(TEST_CSV, "test")
    test_counts = verify_expected_counts(test_df, "test")
    verify_paths_exist(test_df)
    if not args.skip_image_validation:
        validate_image_files(
            test_df,
            args.output_dir / "unreadable_test_images.csv",
            max_failed_images=args.max_failed_images,
        )

    dataset = make_dataset(test_df, batch_size=args.batch_size, training=False, seed=SEED, use_augmentation=False)
    model = tf.keras.models.load_model(args.model)
    real_scores = model.predict(dataset, verbose=1).reshape(-1).astype(float)
    predictions = test_df[["path", "label", "label_str"]].copy()
    predictions = predictions.rename(columns={"label": "true_label", "label_str": "true_label_str"})
    predictions["real_score"] = real_scores
    predictions["fake_score"] = 1.0 - predictions["real_score"]
    predictions["predicted_label"] = (predictions["real_score"] >= 0.5).astype(int)
    predictions["predicted_label_str"] = np.where(predictions["predicted_label"] == 1, "real", "fake")
    predictions["correct"] = predictions["predicted_label"] == predictions["true_label"]
    predictions.to_csv(args.output_dir / "predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    metrics = compute_metrics(predictions)
    metrics["test_counts"] = test_counts
    metrics["model_path"] = str(args.model)
    metrics["model_file_size_bytes"] = int(args.model.stat().st_size)
    metrics["parameter_count"] = int(model.count_params())
    write_json(args.output_dir / "metrics.json", metrics)

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
    plot_training_curves(args.history_csv, args.output_dir)
    write_summary(args.output_dir, args.model, metrics)
    write_comparison(args.model, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
