"""Cortex package entrypoint and lightweight public API."""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Any, Dict

__version__ = "1.0.18"
__author__ = "Cortex Development Team"
__license__ = "MIT"

MINIMUM_PYTHON_VERSION = (3, 11)
SUPPORTED_PLATFORM = "darwin"

_LAZY_EXPORTS = {
    "Config": ("cortex.config", "Config"),
    "GPUValidator": ("cortex.gpu_validator", "GPUValidator"),
    "ModelManager": ("cortex.model_manager", "ModelManager"),
    "InferenceEngine": ("cortex.inference_engine", "InferenceEngine"),
    "ConversationManager": ("cortex.conversation_manager", "ConversationManager"),
}


def verify_system_requirements() -> Dict[str, Any]:
    """Verify that the system meets Cortex requirements."""
    requirements: Dict[str, Any] = {
        "python_version": sys.version_info >= MINIMUM_PYTHON_VERSION,
        "platform": platform.system().lower() == SUPPORTED_PLATFORM,
        "architecture": platform.machine() == "arm64",
        "errors": [],
    }

    if not requirements["python_version"]:
        requirements["errors"].append(
            f"Python {MINIMUM_PYTHON_VERSION[0]}.{MINIMUM_PYTHON_VERSION[1]}+ required, "
            f"found {sys.version_info.major}.{sys.version_info.minor}"
        )

    if not requirements["platform"]:
        requirements["errors"].append(f"macOS required, found {platform.system()}")

    if not requirements["architecture"]:
        requirements["errors"].append(f"ARM64 architecture required, found {platform.machine()}")

    requirements["valid"] = len(requirements["errors"]) == 0
    return requirements


def initialize_cortex() -> bool:
    """Initialize Cortex and verify system compatibility."""
    requirements = verify_system_requirements()

    if not requirements["valid"]:
        for error in requirements["errors"]:
            print(f"❌ {error}", file=sys.stderr)
        return False

    return True


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
    "GPUValidator",
    "ModelManager",
    "InferenceEngine",
    "ConversationManager",
    "initialize_cortex",
    "verify_system_requirements",
]
