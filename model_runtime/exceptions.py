from __future__ import annotations


class ModelRuntimeError(RuntimeError):
    """Base error for model-provider failures."""


class UnknownModelError(ModelRuntimeError):
    """Raised when the active model ID is not present in the registry."""


class DisabledModelError(ModelRuntimeError):
    """Raised when the requested model is present but disabled."""


class ProviderUnavailableError(ModelRuntimeError):
    """Raised when a provider implementation or dependency cannot load."""


class ModelArtifactError(ModelRuntimeError):
    """Raised when the model artifact is missing, invalid, or unexpected."""


class ModelHashError(ModelArtifactError):
    """Raised when model artifact hash verification fails."""


class ModelContractError(ModelRuntimeError):
    """Raised when model input/output contract does not match the registry."""
