"""Shared formatting helpers for slash-command output messages."""

from __future__ import annotations

from typing import Any, Mapping


def _labelize(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def format_key_value_block(*, title: str, values: Mapping[str, Any], include_empty: bool = False) -> str:
    """Format a mapping into a readable multi-line key/value block."""
    lines: list[str] = [title]
    for key in sorted(values.keys()):
        raw_value = values[key]
        if raw_value in (None, "") and not include_empty:
            continue
        value = raw_value if isinstance(raw_value, str) else str(raw_value)
        lines.append(f"- {_labelize(key)}: {value}")
    return "\n".join(lines)


def format_status_summary(status: Mapping[str, Any]) -> str:
    return format_key_value_block(title="System status", values=status)


def format_gpu_status(status: Mapping[str, Any]) -> str:
    return format_key_value_block(title="GPU status", values=status)


def format_auth_status(*, provider: str, auth: Mapping[str, Any]) -> str:
    payload = {"provider": provider, **auth}
    return format_key_value_block(title="Authentication status", values=payload, include_empty=True)
