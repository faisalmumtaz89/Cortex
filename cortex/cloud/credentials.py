"""Credential handling for cloud providers."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from cortex.cloud.types import CloudProvider

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "cortex-llm"

ENV_KEY_MAP: Dict[CloudProvider, str] = {
    CloudProvider.OPENAI: "OPENAI_API_KEY",
    CloudProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    CloudProvider.AZURE: "AZURE_OPENAI_API_KEY",
    CloudProvider.OPENAI_COMPATIBLE: "OPENAI_COMPATIBLE_API_KEY",
}

KEYRING_KEY_MAP: Dict[CloudProvider, str] = {
    CloudProvider.OPENAI: "openai_api_key",
    CloudProvider.ANTHROPIC: "anthropic_api_key",
    CloudProvider.AZURE: "azure_openai_api_key",
    CloudProvider.OPENAI_COMPATIBLE: "openai_compatible_api_key",
}


class CloudCredentialStore:
    """Store and retrieve cloud API keys: an environment variable wins, then
    the system keyring, then a private key file. The file is used where no
    keyring works, such as Linux servers, containers and WSL."""

    def __init__(self, service_name: str = KEYRING_SERVICE, key_file: Optional[Path] = None):
        self.service_name = service_name
        self.key_file = key_file or Path.home() / ".cortex" / "credentials.json"
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

    def _read_key_file(self) -> Dict[str, str]:
        try:
            data = json.loads(self.key_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _write_key_file(self, data: Dict[str, str]) -> None:
        """Replace the key file in one step; it is created readable by its owner only."""
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self.key_file.parent, prefix=".credentials.")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            os.replace(temp, self.key_file)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise

    def get_file_api_key(self, provider: CloudProvider) -> Optional[str]:
        """Return API key from the private key file if present."""
        value = self._read_key_file().get(self._key_name(provider), "").strip()
        return value or None

    def get_api_key_with_source(self, provider: CloudProvider) -> Tuple[Optional[str], Optional[str]]:
        """Get active API key with source indicator."""
        env_value = self.get_env_api_key(provider)
        if env_value:
            return env_value, "env"

        keychain_value = self.get_keychain_api_key(provider)
        if keychain_value:
            return keychain_value, "keychain"

        file_value = self.get_file_api_key(provider)
        if file_value:
            return file_value, "file"

        return None, None

    def get_api_key(self, provider: CloudProvider) -> Optional[str]:
        """Get active API key without source information."""
        key, _ = self.get_api_key_with_source(provider)
        return key

    def get_auth_summary(self, provider: CloudProvider) -> Dict[str, object]:
        """Return a summary for login/status UIs."""
        env_value = self.get_env_api_key(provider)
        keychain_value = self.get_keychain_api_key(provider)
        file_value = self.get_file_api_key(provider)
        _, active_source = self.get_api_key_with_source(provider)

        return {
            "active_source": active_source,
            "env_present": bool(env_value),
            "keychain_present": bool(keychain_value),
            "file_present": bool(file_value),
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

        if self._keyring is not None:
            try:
                self._keyring.set_password(self.service_name, self._key_name(provider), value)
            except Exception as exc:
                logger.info("No usable system keyring (%s); saving the key to %s", exc, self.key_file)
            else:
                self._forget_file_key(provider)
                return True, "API key saved to keychain."

        try:
            data = self._read_key_file()
            data[self._key_name(provider)] = value
            self._write_key_file(data)
        except OSError as exc:
            return (
                False,
                f"Failed to save key: {exc}. Set {self._env_name(provider)} instead.",
            )
        return True, f"API key saved to {self.key_file} (readable only by you)."

    def _forget_file_key(self, provider: CloudProvider) -> bool:
        data = self._read_key_file()
        if self._key_name(provider) not in data:
            return False
        del data[self._key_name(provider)]
        try:
            self._write_key_file(data)
        except OSError:
            return False
        return True

    def delete_api_key(self, provider: CloudProvider) -> Tuple[bool, str]:
        """Delete the provider's saved key from the keychain and the key file."""
        from_file = self._forget_file_key(provider)
        if self._keyring is None:
            return (True, "Saved key removed.") if from_file else (False, "No saved key found.")

        try:
            self._keyring.delete_password(self.service_name, self._key_name(provider))
        except self._keyring_delete_error:
            if from_file:
                return True, "Saved key removed."
            return False, "No saved key found in keychain."
        except Exception as exc:
            if from_file:
                return True, "Saved key removed."
            return False, f"Failed to delete key: {exc}"

        return True, "Saved key removed from keychain."
