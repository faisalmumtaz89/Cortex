"""Cortex package entrypoint and lightweight public API."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "1.0.19"
__author__ = "Cortex Development Team"
__license__ = "MIT"

_LAZY_EXPORTS = {
    "Config": ("cortex.config", "Config"),
    "ConversationManager": ("cortex.conversation_manager", "ConversationManager"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "Config",
    "ConversationManager",
]
