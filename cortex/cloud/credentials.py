"""Credential handling for cloud providers."""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

from cortex.cloud.types import CloudProvider

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "cortex-llm"

ENV_KEY_MAP: Dict[CloudProvider, str] = {
    CloudProvider.OPENAI: "OPENAI_API_KEY",
    CloudProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
}

KEYRING_KEY_MAP: Dict[CloudProvider, str] = {
    CloudProvider.OPENAI: "openai_api_key",
    CloudProvider.ANTHROPIC: "anthropic_api_key",
}


class CloudCredentialStore:
    """Store and retrieve cloud API keys."""

    def __init__(self, service_name: str = KEYRING_SERVICE):
        self.service_name = service_name
        self._keyring = None
        self._keyring_delete_error = Exception

        try:
            import keyring  # type: ignore
            from keyring.errors import PasswordDeleteError  # type: ignore

            self._keyring = keyring
            self._keyring_delete_error = PasswordDeleteError
        except Exception:
            self._keyring = None

    def _env_name(self, provider: CloudProvider) -> str:
        return ENV_KEY_MAP[provider]

    def _key_name(self, provider: CloudProvider) -> str:
        return KEYRING_KEY_MAP[provider]

    def get_env_api_key(self, provider: CloudProvider) -> Optional[str]:
        """Return API key from environment if present."""
        value = os.getenv(self._env_name(provider), "").strip()
        return value or None

    def get_keychain_api_key(self, provider: CloudProvider) -> Optional[str]:
        """Return API key from keychain if present."""
        if self._keyring is None:
            return None
        try:
            value = self._keyring.get_password(self.service_name, self._key_name(provider))
        except Exception as exc:
            logger.debug("Failed to read keychain credential for %s: %s", provider.value, exc)
            return None
        if value is None:
            return None
        value = value.strip()
        return value or None

    def get_api_key_with_source(self, provider: CloudProvider) -> Tuple[Optional[str], Optional[str]]:
        """Get active API key with source indicator."""
        env_value = self.get_env_api_key(provider)
        if env_value:
            return env_value, "env"

        keychain_value = self.get_keychain_api_key(provider)
        if keychain_value:
            return keychain_value, "keychain"

        return None, None

    def get_api_key(self, provider: CloudProvider) -> Optional[str]:
        """Get active API key without source information."""
        key, _ = self.get_api_key_with_source(provider)
        return key

    def get_auth_summary(self, provider: CloudProvider) -> Dict[str, object]:
        """Return a summary for login/status UIs."""
        env_value = self.get_env_api_key(provider)
        keychain_value = self.get_keychain_api_key(provider)
        active_source = None
        if env_value:
            active_source = "env"
        elif keychain_value:
            active_source = "keychain"

        return {
            "active_source": active_source,
            "env_present": bool(env_value),
            "keychain_present": bool(keychain_value),
            "env_var": self._env_name(provider),
        }

    def is_authenticated(self, provider: CloudProvider) -> bool:
        """True when any usable key exists."""
        return self.get_api_key(provider) is not None

    def save_api_key(self, provider: CloudProvider, api_key: str) -> Tuple[bool, str]:
        """Save API key in keychain."""
        value = api_key.strip()
        if not value:
            return False, "API key cannot be empty."

        if self._keyring is None:
            return (
                False,
                "Keyring unavailable. Use environment variable "
                f"{self._env_name(provider)} instead.",
            )

        try:
            self._keyring.set_password(self.service_name, self._key_name(provider), value)
        except Exception as exc:
            return False, f"Failed to save key: {exc}"
        return True, "API key saved to keychain."

    def delete_api_key(self, provider: CloudProvider) -> Tuple[bool, str]:
        """Delete provider key from keychain."""
        if self._keyring is None:
            return False, "Keyring unavailable."

        try:
            self._keyring.delete_password(self.service_name, self._key_name(provider))
        except self._keyring_delete_error:
            return False, "No saved key found in keychain."
        except Exception as exc:
            return False, f"Failed to delete key: {exc}"

        return True, "Saved key removed from keychain."
