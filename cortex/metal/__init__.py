"""Metal optimization package for GPU acceleration on Apple Silicon.

This package provides unified GPU acceleration for LLM inference on Apple Silicon.
The recommended approach is to use MetalOptimizer for automatic backend selection.
"""

from typing import Dict, Any, Optional
import platform
import subprocess

# Primary exports
__all__ = [
    # Unified optimizer (RECOMMENDED)
    "MetalOptimizer",
    "OptimizationConfig",
    "Backend",
    "InferenceSession",
    
    # Core functionality
    "MetalCapabilities",
    "check_metal_support",
    "get_metal_version",
    "initialize_metal_optimizations",
    
    # Memory management
    "MemoryPool",
    
    # Backend-specific (use MetalOptimizer instead for most cases)
    "MPSOptimizer",
    "MLXAccelerator",
    
    # Performance monitoring
    "PerformanceProfiler"
]

class MetalCapabilities:
    """Metal capabilities detection and management."""
    
    METAL_FEATURES = {
        "metal3": {
            "min_macos": "14.0",
            "features": [
                "mesh_shaders",
                "function_pointers",
                "ray_tracing",
                "indirect_command_buffers",
                "gpu_driven_pipeline"
            ]
        },
        "metal2": {
            "min_macos": "10.13",
            "features": [
                "argument_buffers",
                "programmable_sample_positions",
                "texture_read_write"
            ]
        }
    }
    
    APPLE_SILICON_OPTIMIZATION_FLAGS = {
        "compiler_flags": [
            "-O3",
            "-ffast-math",
            "-march=armv8.5-a+fp16+dotprod",
            "-mtune=apple-silicon"
        ],
        "metal_compiler_flags": [
            # Use macOS-appropriate Metal standard version
            "-std=metal3.1",
            "-O3",
            "-ffast-math"
        ],
        "linker_flags": [
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
            "-framework", "MetalPerformanceShadersGraph"
        ]
    }
    
    @classmethod
    def detect_capabilities(cls) -> Dict[str, Any]:
        """Detect Metal capabilities on the system."""
        if platform.system() != "Darwin":
            return {"supported": False, "error": "Not running on macOS"}
        
        capabilities = {
            "supported": True,
            "version": get_metal_version(),
            "features": [],
            "optimizations": {},
            "gpu_family": cls._detect_gpu_family()
        }
        
        metal_version = capabilities["version"]
        if metal_version and "Metal 3" in metal_version:
            capabilities["features"] = cls.METAL_FEATURES["metal3"]["features"]
        elif metal_version and "Metal 2" in metal_version:
            capabilities["features"] = cls.METAL_FEATURES["metal2"]["features"]
        
        # Always assign optimization profile based on detected GPU family
        capabilities["optimizations"] = cls.get_optimization_profile(capabilities["gpu_family"])
        
        return capabilities
    
    @classmethod
    def _detect_gpu_family(cls) -> str:
        """Detect GPU family (apple5, apple6, apple7, apple8 for M1, M2, M3, M4)."""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                check=True
            )
            
            output = result.stdout.lower()
            if "apple m4" in output:
                return "apple8"
            elif "apple m3" in output:
                return "apple7"
            elif "apple m2" in output:
                return "apple6"
            elif "apple m1" in output:
                return "apple5"
            else:
                return "unknown"
        except:
            return "unknown"
    
    @classmethod
    def get_optimization_profile(cls, gpu_family: str) -> Dict[str, Any]:
        """Get optimization profile for specific GPU family."""
        profiles = {
            "apple8": {  # M4
                "max_threads_per_threadgroup": 1024,
                "max_total_threadgroup_memory": 32768,
                "simd_width": 32,
                "preferred_batch_size": 8,
                "use_fused_operations": True,
                "use_fast_math": True,
                "tile_size": (16, 16),
                "wave_size": 32,
                "prefer_bfloat16": True,
            },
            "apple7": {  # M3
                "max_threads_per_threadgroup": 1024,
                "max_total_threadgroup_memory": 32768,
                "simd_width": 32,
                "preferred_batch_size": 4,
                "use_fused_operations": True,
                "use_fast_math": True,
                "tile_size": (8, 8),
                "wave_size": 32,
                "prefer_bfloat16": True,
            },
            "apple6": {  # M2
                "max_threads_per_threadgroup": 1024,
                "max_total_threadgroup_memory": 32768,
                "simd_width": 32,
                "preferred_batch_size": 4,
                "use_fused_operations": True,
                "use_fast_math": True,
                "tile_size": (8, 8),
                "wave_size": 32,
                "prefer_bfloat16": True,
            },
            "apple5": {  # M1
                "max_threads_per_threadgroup": 1024,
                "max_total_threadgroup_memory": 32768,
                "simd_width": 32,
                "preferred_batch_size": 2,
                "use_fused_operations": False,
                "use_fast_math": True,
                "tile_size": (8, 8),
                "wave_size": 32,
                "prefer_bfloat16": False,
            },
            "default": {
                "max_threads_per_threadgroup": 512,
                "max_total_threadgroup_memory": 16384,
                "simd_width": 32,
                "preferred_batch_size": 2,
                "use_fused_operations": False,
                "use_fast_math": False,
                "tile_size": (8, 8),
                "wave_size": 32,
                "prefer_bfloat16": False,
            }
        }
        
        return profiles.get(gpu_family, profiles["default"])

def check_metal_support() -> bool:
    """Check if Metal is supported on this system."""
    if platform.system() != "Darwin":
        return False
    
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            check=True
        )
        return "Metal" in result.stdout
    except:
        return False

def get_metal_version() -> Optional[str]:
    """Get Metal version string."""
    try:
        result = subprocess.run(
            ["xcrun", "--show-sdk-version"],
            capture_output=True,
            text=True,
            check=True
        )
        sdk_version = result.stdout.strip()
        
        major_version = int(sdk_version.split('.')[0])
        if major_version >= 14:
            return "Metal 3"
        elif major_version >= 10:
            return "Metal 2"
        else:
            return "Metal 1"
    except:
        return None

def initialize_metal_optimizations() -> Dict[str, Any]:
    """Initialize Metal optimizations for the current system."""
    if not check_metal_support():
        raise RuntimeError("Metal is not supported on this system")
    
    capabilities = MetalCapabilities.detect_capabilities()
    
    if not capabilities["supported"]:
        raise RuntimeError(f"Metal not supported: {capabilities.get('error', 'Unknown error')}")
    
    gpu_family = capabilities["gpu_family"]
    optimization_profile = MetalCapabilities.get_optimization_profile(gpu_family)
    
    return {
        "capabilities": capabilities,
        "optimization_profile": optimization_profile,
        "gpu_family": gpu_family,
        "metal_version": capabilities["version"]
    }

# Import new unified optimizer (RECOMMENDED)
try:
    from cortex.metal.optimizer import (
        MetalOptimizer,
        OptimizationConfig,
        Backend,
        InferenceSession
    )
except ImportError as e:
    # If optimizer fails to import, provide a helpful message
    import warnings
    warnings.warn(
        f"Failed to import MetalOptimizer: {e}. "
        "Some dependencies may be missing.",
        ImportWarning
    )
    MetalOptimizer = None
    OptimizationConfig = None
    Backend = None
    InferenceSession = None

# Import existing components
from cortex.metal.memory_pool import MemoryPool
from cortex.metal.mps_optimizer import MPSOptimizer
from cortex.metal.mlx_accelerator import MLXAccelerator
from cortex.metal.performance_profiler import PerformanceProfiler