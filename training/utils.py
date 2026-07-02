from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataframe_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label_str"].value_counts().to_dict()
    return {str(key): int(value) for key, value in sorted(counts.items())}


def plot_training_curves(history_csv: Path, output_dir: Path) -> None:
    if not history_csv.exists():
        return
    history = pd.read_csv(history_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    def plot_pair(train_col: str, val_col: str, title: str, filename: str) -> None:
        if train_col not in history.columns or val_col not in history.columns:
            return
        plt.figure(figsize=(7, 4.5))
        plt.plot(history["epoch"], history[train_col], label=train_col)
        plt.plot(history["epoch"], history[val_col], label=val_col)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()

    plot_pair("loss", "val_loss", "Loss", "loss_curve.png")
    plot_pair("accuracy", "val_accuracy", "Accuracy", "accuracy_curve.png")
    plot_pair("auc", "val_auc", "ROC-AUC", "auc_curve.png")


def model_summary_text(model: tf.keras.Model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines) + "\n"


def append_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
