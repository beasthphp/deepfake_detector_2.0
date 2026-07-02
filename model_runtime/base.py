from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from model_runtime.metadata import ModelMetadata


class DeepfakeModelProvider(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Stable metadata exposed to API clients."""

    @abstractmethod
    def load(self) -> None:
        """Load the underlying model artifact."""

    @abstractmethod
    def validate(self) -> None:
        """Validate file, provider, and input/output contract."""

    @abstractmethod
    def predict_batch(self, images: list[Any]) -> list[dict[str, float | str]]:
        """Return real_score, fake_score, and label for each image."""
