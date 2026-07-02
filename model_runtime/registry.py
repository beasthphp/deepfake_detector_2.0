from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_runtime.base import DeepfakeModelProvider
from model_runtime.exceptions import DisabledModelError, ProviderUnavailableError, UnknownModelError
from model_runtime.metadata import ModelMetadata, ScoreThresholds, ThresholdProfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = ROOT / "models" / "registry.json"
DEFAULT_ARTIFACT_DIR = ROOT / "models" / "artifacts"

PROVIDERS = {
    "tensorflow_binary_face": "model_runtime.providers.tensorflow_binary_face.TensorFlowBinaryFaceProvider",
}


@dataclass(frozen=True)
class RegistryModel:
    id: str
    version: str
    framework: str
    provider: str
    local_filename: str
    download_url: str | None
    sha256: str | None
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]
    output_meaning: str
    class_mapping: dict[str, int]
    threshold_profile: ThresholdProfile
    enabled: bool
    file_types: tuple[str, ...]
    model_name: str
    raw: dict[str, Any]

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            id=self.id,
            version=self.version,
            framework=self.framework,
            provider=self.provider,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            output_meaning=self.output_meaning,
            class_mapping=self.class_mapping,
            threshold_profile=self.threshold_profile,
            model_name=self.model_name,
        )


@dataclass(frozen=True)
class ModelRegistry:
    path: Path
    default_model_id: str
    models: dict[str, RegistryModel]


def load_registry(registry_path: str | Path | None = None) -> ModelRegistry:
    path = _resolve_path(
        registry_path or os.getenv("MODEL_REGISTRY_PATH") or DEFAULT_REGISTRY_PATH,
        base=ROOT,
    )
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    models = {}
    for item in data.get("models", []):
        model = _parse_model(item)
        models[model.id] = model
    default_model_id = data.get("default_model_id") or os.getenv("ACTIVE_MODEL_ID") or "legacy-cnn-v1"
    return ModelRegistry(path=path, default_model_id=default_model_id, models=models)


def get_registry_model(
    active_model_id: str | None = None,
    registry_path: str | Path | None = None,
) -> RegistryModel:
    registry = load_registry(registry_path)
    model_id = active_model_id or os.getenv("ACTIVE_MODEL_ID") or registry.default_model_id
    if model_id not in registry.models:
        raise UnknownModelError(f"Active model is unknown: {model_id}")
    model = registry.models[model_id]
    if not model.enabled:
        raise DisabledModelError(f"Active model is disabled in registry: {model_id}")
    return model


def create_provider(
    active_model_id: str | None = None,
    registry_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    explicit_model_path: str | Path | None = None,
) -> DeepfakeModelProvider:
    registry = load_registry(registry_path)
    model_id = active_model_id or os.getenv("ACTIVE_MODEL_ID") or registry.default_model_id
    if model_id not in registry.models:
        raise UnknownModelError(f"Active model is unknown: {model_id}")
    entry = registry.models[model_id]
    if not entry.enabled:
        raise DisabledModelError(f"Active model is disabled in registry: {model_id}")

    provider_type = PROVIDERS.get(entry.provider)
    if provider_type is None:
        raise ProviderUnavailableError(f"Provider is not registered: {entry.provider}")

    module_name, class_name = provider_type.rsplit(".", 1)
    try:
        provider_class = getattr(importlib.import_module(module_name), class_name)
    except Exception as exc:
        raise ProviderUnavailableError(f"Provider is unavailable: {entry.provider}") from exc

    model_path = _resolve_model_path(entry, artifact_dir=artifact_dir, explicit_model_path=explicit_model_path)
    return provider_class(entry=entry, model_path=model_path)


def _resolve_model_path(
    entry: RegistryModel,
    artifact_dir: str | Path | None = None,
    explicit_model_path: str | Path | None = None,
) -> Path:
    override = explicit_model_path or os.getenv("DEEPFAKE_MODEL_PATH")
    if override:
        return _resolve_path(override, base=ROOT)
    artifact_root = _resolve_path(
        artifact_dir or os.getenv("MODEL_ARTIFACT_DIR") or DEFAULT_ARTIFACT_DIR,
        base=ROOT,
    )
    return artifact_root / entry.local_filename


def _parse_model(item: dict[str, Any]) -> RegistryModel:
    threshold = item.get("threshold_profile", {})
    thresholds = ScoreThresholds(
        likely_fake_below=float(threshold.get("likely_fake_below", 0.40)),
        likely_real_above=float(threshold.get("likely_real_above", 0.60)),
    )
    return RegistryModel(
        id=str(item["id"]),
        version=str(item["version"]),
        framework=str(item["framework"]),
        provider=str(item["provider"]),
        local_filename=str(item["local_filename"]),
        download_url=item.get("download_url") or None,
        sha256=item.get("sha256") or None,
        input_shape=_parse_shape(item.get("input_size") or item.get("input_shape")),
        output_shape=_parse_shape(item.get("output_shape", [None, 1])),
        output_meaning=str(item.get("output_interpretation", "")),
        class_mapping={str(key): int(value) for key, value in item.get("class_mapping", {}).items()},
        threshold_profile=ThresholdProfile(
            id=str(threshold.get("id", "score-thresholds")),
            version=str(threshold.get("version", "1")),
            thresholds=thresholds,
        ),
        enabled=bool(item.get("enabled", False)),
        file_types=tuple(str(value).lower() for value in item.get("file_types", [".h5", ".keras"])),
        model_name=str(item.get("name", "human-face deepfake classifier")),
        raw=item,
    )


def _parse_shape(value: Any) -> tuple[int | None, ...]:
    if value is None:
        return (None, 256, 256, 3)
    if len(value) == 3:
        return (None, int(value[0]), int(value[1]), int(value[2]))
    return tuple(None if item is None else int(item) for item in value)


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()
