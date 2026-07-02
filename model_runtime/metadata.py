from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreThresholds:
    likely_fake_below: float = 0.40
    likely_real_above: float = 0.60


@dataclass(frozen=True)
class ThresholdProfile:
    id: str = "score-thresholds"
    version: str = "1"
    thresholds: ScoreThresholds = ScoreThresholds()


@dataclass(frozen=True)
class ModelMetadata:
    id: str
    version: str
    framework: str
    provider: str
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]
    output_meaning: str
    class_mapping: dict[str, int]
    threshold_profile: ThresholdProfile
    model_name: str = "human-face deepfake classifier"
