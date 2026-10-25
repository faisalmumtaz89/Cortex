"""Shared cloud model types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


class CloudProvider(str, Enum):
    """Supported cloud providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    OPENAI_COMPATIBLE = "openai-compatible"

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
    """The model the next turn runs on, if one is selected."""

    cloud_model: Optional[CloudModelRef] = None

    @classmethod
    def cloud(cls, model_ref: CloudModelRef) -> "ActiveModelTarget":
        """Build a target for a cloud model."""
        return cls(cloud_model=model_ref)

    @property
    def label(self) -> str:
        """Human-readable label for status displays."""
        return self.cloud_model.selector if self.cloud_model else "No model loaded"
