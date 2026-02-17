"""Shared cloud model types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union


class CloudProvider(str, Enum):
    """Supported cloud providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"

    @classmethod
    def from_value(cls, value: Union[str, "CloudProvider"]) -> "CloudProvider":
        """Convert a raw value to a provider enum."""
        if isinstance(value, cls):
            return value

        normalized = str(value).strip().lower()
        for provider in cls:
            if provider.value == normalized:
                return provider
        raise ValueError(f"Unsupported cloud provider: {value}")


@dataclass(frozen=True)
class CloudModelRef:
    """Provider + model identifier."""

    provider: CloudProvider
    model_id: str

    @property
    def selector(self) -> str:
        """Provider-prefixed selector representation."""
        return f"{self.provider.value}:{self.model_id}"


@dataclass
class ActiveModelTarget:
    """Current active model target for generation."""

    backend: Literal["local", "cloud"] = "local"
    local_model: Optional[str] = None
    cloud_model: Optional[CloudModelRef] = None

    @classmethod
    def local(cls, model_name: Optional[str] = None) -> "ActiveModelTarget":
        """Build a local target."""
        return cls(backend="local", local_model=model_name, cloud_model=None)

    @classmethod
    def cloud(cls, model_ref: CloudModelRef) -> "ActiveModelTarget":
        """Build a cloud target."""
        return cls(backend="cloud", local_model=None, cloud_model=model_ref)

    @property
    def label(self) -> str:
        """Human-readable label for status displays."""
        if self.backend == "cloud" and self.cloud_model:
            return self.cloud_model.selector
        return self.local_model or "No model loaded"
