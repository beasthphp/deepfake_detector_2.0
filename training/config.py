from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "data" / "raw"
IMAGE_ROOT = DATA_ROOT / "real_vs_fake" / "real-vs-fake"

TRAIN_CSV = DATA_ROOT / "train.csv"
VALID_CSV = DATA_ROOT / "valid.csv"
TEST_CSV = DATA_ROOT / "test.csv"

EXISTING_MODEL_PATH = ROOT / "model" / "deepfake_detector_93acc.h5"
ORIGINAL_MODEL_NOTE = ROOT / "models" / "original" / "README.md"

EXPERIMENT_MODEL_DIR = ROOT / "models" / "experiments"
RETRAINED_BEST_MODEL = EXPERIMENT_MODEL_DIR / "retrained_custom_cnn_best.keras"
SMOKE_BEST_MODEL = EXPERIMENT_MODEL_DIR / "retrained_custom_cnn_smoke_best.keras"

REPORT_DIR = ROOT / "reports" / "retrained_baseline"
COMPARISON_REPORT = ROOT / "reports" / "BASELINE_COMPARISON.md"
EXISTING_MODEL_METRICS = ROOT / "reports" / "existing_model" / "metrics.json"

IMAGE_SIZE = (256, 256)
CHANNELS = 3
CLASS_NAMES = ["fake", "real"]
CLASS_TO_LABEL = {"fake": 0, "real": 1}
LABEL_TO_CLASS = {0: "fake", 1: "real"}

EXPECTED_COUNTS = {
    "train": {"fake": 50_000, "real": 50_000},
    "valid": {"fake": 10_000, "real": 10_000},
    "test": {"fake": 10_000, "real": 10_000},
}

SEED = 69


@dataclass(frozen=True)
class BaselineTrainingConfig:
    batch_size: int = 32
    max_epochs: int = 15
    learning_rate: float = 0.001
    early_stopping_patience: int = 3
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    max_failed_images: int = 0
    smoke_train_per_class: int = 500
    smoke_valid_per_class: int = 100


DEFAULT_TRAINING = BaselineTrainingConfig()
