"""Shared formatting helpers for slash-command output messages."""

from __future__ import annotations

from typing import Any, Mapping

# Exact labels for keys whose acronyms/units a blanket title-case would mangle
# (e.g. "gpu_cores" -> "Gpu cores", "total_memory_gb" -> "Total memory gb").
_KEY_LABELS: dict[str, str] = {
    "gpu": "GPU",
    "gpu_cores": "GPU cores",
    "gpu_acceleration": "GPU acceleration",
    "gpu_utilization": "GPU utilization",
    "mlx": "MLX",
    "mlx_available": "MLX available",
    "mlx_acceleration": "MLX acceleration",
    "mps": "MPS",
    "mps_available": "MPS available",
    "metal": "Metal",
    "metal_available": "Metal available",
    "cpu": "CPU",
    "os": "OS",
    "id": "ID",
    "url": "URL",
    "api": "API",
    "total_memory_gb": "Total memory (GB)",
    "available_memory_gb": "Available (GB)",
    "used_memory_gb": "Used (GB)",
    "memory_gb": "Memory (GB)",
    "kv_cache_gb": "KV cache (GB)",
    "pool_size_gb": "Pool size (GB)",
}

# Tokens that should stay uppercase when they appear as a standalone word in an
# otherwise title-cased label.
_ACRONYMS = {"gpu", "mlx", "mps", "cpu", "os", "id", "url", "api", "gb", "mb", "kb", "kv", "ram"}


def _labelize(key: str) -> str:
    exact = _KEY_LABELS.get(key.strip().lower())
    if exact is not None:
        return exact
    # Sentence case with acronym words kept uppercase: "gpu_cores" -> "GPU cores",
    # "active_model" -> "Active model", "chip_name" -> "Chip name".
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


def format_gpu_status(status: Mapping[str, Any]) -> str:
    return format_key_value_block(title="GPU status", values=status)


def format_auth_status(*, provider: str, auth: Mapping[str, Any]) -> str:
    payload = {"provider": provider, **auth}
    return format_key_value_block(title="Authentication status", values=payload, include_empty=True)
