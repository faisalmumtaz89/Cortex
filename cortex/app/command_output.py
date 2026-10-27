"""Shared formatting helpers for slash-command output messages."""

from __future__ import annotations

from typing import Any, Mapping

# Tokens that should stay uppercase when they appear as a standalone word in an
# otherwise title-cased label.
_ACRONYMS = {"cpu", "os", "id", "url", "api", "gb", "mb", "kb", "ram"}


def _labelize(key: str) -> str:
    # Sentence case with acronym words kept uppercase: "api_key" -> "API key",
    # "active_model" -> "Active model".
    words = [
        word.upper() if word.lower() in _ACRONYMS else word.lower()
        for word in key.replace("_", " ").strip().split()
    ]
    label = " ".join(words)
    return label[:1].upper() + label[1:] if label else label


def format_key_value_block(*, title: str, values: Mapping[str, Any], include_empty: bool = False) -> str:
    """Format a mapping into a readable multi-line key/value block.

    Field order follows the mapping's insertion order (callers curate it); it is
    never alphabetized.
    """
    lines: list[str] = [title]
    for key, raw_value in values.items():
        if raw_value in (None, "") and not include_empty:
            continue
        value = raw_value if isinstance(raw_value, str) else str(raw_value)
        lines.append(f"- {_labelize(key)}: {value}")
    return "\n".join(lines)


def format_status_summary(status: Mapping[str, Any]) -> str:
    return format_key_value_block(title="System status", values=status)


def format_auth_status(*, provider: str, auth: Mapping[str, Any]) -> str:
    """Where the key /login status reports comes from, as sentences."""
    env_var = str(auth.get("env_var") or "")
    source = auth.get("active_source")
    saved = bool(auth.get("keychain_present") or auth.get("file_present"))
    where = {
        "env": f"the {env_var} environment variable",
        "keychain": "the key saved with /login, in the system keyring",
        "file": "the key saved with /login, in ~/.cortex/credentials.json",
    }
    if source in where:
        message = f"{provider}: logged in, using {where[str(source)]}."
        if source == "env" and saved:
            message += f" A key saved with /login is also there; {env_var} takes precedence."
        return message
    return f"{provider}: not logged in. Add a key with /login {provider} <api_key>, or set {env_var}."
