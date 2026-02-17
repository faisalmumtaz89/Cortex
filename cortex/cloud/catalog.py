"""Cloud model catalog and selector parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from cortex.cloud.types import CloudModelRef, CloudProvider

DEFAULT_MODELS: Dict[CloudProvider, List[str]] = {
    CloudProvider.OPENAI: [
        "gpt-5.1",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    CloudProvider.ANTHROPIC: [
        "claude-opus-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ],
}


class CloudModelCatalog:
    """Curated cloud model catalog with optional user overrides."""

    def __init__(self, override_path: Optional[Path] = None):
        self.override_path = (override_path or (Path.home() / ".cortex" / "cloud_models.json")).expanduser()

    def _read_override_file(self) -> Dict[str, List[str]]:
        """Read model overrides from disk."""
        if not self.override_path.exists():
            return {}

        try:
            with open(self.override_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}

        if not isinstance(payload, dict):
            return {}

        normalized: Dict[str, List[str]] = {}
        for key, value in payload.items():
            if not isinstance(value, list):
                continue
            entries = [str(item).strip() for item in value if str(item).strip()]
            if entries:
                normalized[str(key).strip().lower()] = entries
        return normalized

    def _merge_models(self, provider: CloudProvider) -> List[str]:
        """Merge default and overridden models for provider."""
        base = list(DEFAULT_MODELS.get(provider, []))
        overrides = self._read_override_file().get(provider.value, [])

        merged: List[str] = []
        for model_id in [*base, *overrides]:
            if model_id not in merged:
                merged.append(model_id)
        return merged

    def list_model_ids(self, provider: CloudProvider) -> List[str]:
        """List model ids for a provider."""
        return self._merge_models(provider)

    def list_models(self, provider: Optional[CloudProvider] = None) -> List[CloudModelRef]:
        """List cloud models, optionally filtered by provider."""
        providers: Sequence[CloudProvider]
        if provider is None:
            providers = [CloudProvider.OPENAI, CloudProvider.ANTHROPIC]
        else:
            providers = [provider]

        refs: List[CloudModelRef] = []
        for item in providers:
            for model_id in self.list_model_ids(item):
                refs.append(CloudModelRef(provider=item, model_id=model_id))
        return refs

    def has_model(self, provider: CloudProvider, model_id: str) -> bool:
        """Return True when model exists in curated catalog."""
        return model_id in self.list_model_ids(provider)

    def parse_selector(self, raw: str) -> Optional[CloudModelRef]:
        """Parse selector in provider:model format."""
        if ":" not in raw:
            return None

        provider_raw, model_raw = raw.split(":", 1)
        provider_raw = provider_raw.strip().lower()
        model_raw = model_raw.strip()
        if not provider_raw or not model_raw:
            return None

        try:
            provider = CloudProvider.from_value(provider_raw)
        except ValueError:
            return None

        return CloudModelRef(provider=provider, model_id=model_raw)

    def ensure_model(self, provider: Union[str, CloudProvider], model_id: str) -> CloudModelRef:
        """Return normalized model reference."""
        return CloudModelRef(provider=CloudProvider.from_value(provider), model_id=model_id.strip())

    def parse_or_raise(self, raw: str) -> Tuple[CloudProvider, str]:
        """Parse a provider:model selector or raise ValueError."""
        ref = self.parse_selector(raw)
        if ref is None:
            raise ValueError(f"Invalid cloud model selector: {raw}")
        return ref.provider, ref.model_id
