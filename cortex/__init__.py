"""
Cortex - GPU-Accelerated LLM Terminal for Apple Silicon

A high-performance terminal interface for running Hugging Face LLMs locally
with exclusive GPU acceleration via Metal Performance Shaders (MPS) and MLX.
"""

__version__ = "1.0.7"
__author__ = "Cortex Development Team"
__license__ = "MIT"

from typing import Optional, Dict, Any
import platform
import sys

MINIMUM_PYTHON_VERSION = (3, 11)
SUPPORTED_PLATFORM = "darwin"

def verify_system_requirements() -> Dict[str, Any]:
    """Verify that the system meets Cortex requirements."""
    requirements = {
        "python_version": sys.version_info >= MINIMUM_PYTHON_VERSION,
        "platform": platform.system().lower() == SUPPORTED_PLATFORM,
        "architecture": platform.machine() == "arm64",
        "errors": []
    }
    
    if not requirements["python_version"]:
        requirements["errors"].append(
            f"Python {MINIMUM_PYTHON_VERSION[0]}.{MINIMUM_PYTHON_VERSION[1]}+ required, "
            f"found {sys.version_info.major}.{sys.version_info.minor}"
        )
    
    if not requirements["platform"]:
        requirements["errors"].append(
            f"macOS required, found {platform.system()}"
        )
    
    if not requirements["architecture"]:
        requirements["errors"].append(
            f"ARM64 architecture required, found {platform.machine()}"
        )
    
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

from cortex.config import Config
from cortex.gpu_validator import GPUValidator
from cortex.model_manager import ModelManager
from cortex.inference_engine import InferenceEngine
from cortex.conversation_manager import ConversationManager

__all__ = [
    "__version__",
    "Config",
    "GPUValidator",
    "ModelManager",
    "InferenceEngine",
    "ConversationManager",
    "initialize_cortex",
    "verify_system_requirements"
]
