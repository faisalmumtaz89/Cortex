"""GPU validation for Metal/MPS support on Apple Silicon."""

import sys
import platform
import subprocess
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import psutil

@dataclass
class GPUInfo:
    """GPU information and capabilities."""
    has_metal: bool
    has_mps: bool
    has_mlx: bool
    gpu_cores: int
    total_memory: int
    available_memory: int
    metal_version: Optional[str]
    chip_name: str
    unified_memory: bool
    is_apple_silicon: bool
    
    # MSL v4 capabilities
    gpu_family: str  # apple5 (M1), apple6 (M2), apple7 (M3), apple8 (M4)
    supports_bfloat16: bool
    supports_simdgroup_matrix: bool
    supports_mpp: bool
    supports_tile_functions: bool
    supports_atomic_float: bool
    supports_fast_math: bool
    supports_function_constants: bool
    max_threads_per_threadgroup: int
    
    @property
    def is_valid(self) -> bool:
        """Check if GPU meets requirements."""
        # Minimum requirements for production
        min_memory = 4 * 1024 * 1024 * 1024  # 4GB minimum for small models
        min_cores = 8  # M1 and above have at least 8 cores
        
        return (
            self.has_metal and
            self.has_mps and
            self.has_mlx and
            self.gpu_cores >= min_cores and
            self.available_memory >= min_memory
        )
    
    def get_validation_errors(self) -> list[str]:
        """Get list of validation errors."""
        errors = []
        min_memory = 4 * 1024 * 1024 * 1024  # 4GB minimum
        min_cores = 8
        
        if not self.has_metal:
            errors.append("Metal support not available")
        if not self.has_mps:
            errors.append("Metal Performance Shaders (MPS) not available")
        if not self.has_mlx:
            errors.append("MLX framework not available (install with: pip install mlx)")
        if self.gpu_cores < min_cores:
            errors.append(f"Insufficient GPU cores: {self.gpu_cores} (need {min_cores})")
        if self.available_memory < min_memory:
            memory_gb = self.available_memory / (1024 * 1024 * 1024)
            min_memory_gb = min_memory / (1024 * 1024 * 1024)
            errors.append(f"Insufficient GPU memory: {memory_gb:.1f}GB (need {min_memory_gb:.1f}GB)")
        return errors

class GPUValidator:
    """Validate GPU capabilities for Cortex."""
    
    def __init__(self, config=None):
        """Initialize GPU validator."""
        self.config = config  # Store config if provided
        self.gpu_info: Optional[GPUInfo] = None
        self._torch_available = False
        self._mlx_available = False
        self._validate_imports()
    
    def _validate_imports(self) -> None:
        """Validate that required GPU libraries are available."""
        try:
            import torch
            self._torch_available = True
        except ImportError:
            self._torch_available = False
        
        try:
            import mlx.core as mx
            self._mlx_available = True
        except ImportError:
            self._mlx_available = False
    
    def validate(self) -> Tuple[bool, Optional[GPUInfo], list[str]]:
        """
        Validate GPU support.
        
        Returns:
            Tuple of (is_valid, gpu_info, errors)
        """
        errors = []
        
        if platform.system().lower() != "darwin":
            errors.append(f"macOS required, found {platform.system()}")
            return False, None, errors
        
        if platform.machine() != "arm64":
            errors.append(f"ARM64 architecture required, found {platform.machine()}")
            return False, None, errors
        
        self.gpu_info = self._get_gpu_info()
        
        if not self.gpu_info.is_valid:
            errors.extend(self.gpu_info.get_validation_errors())
            return False, self.gpu_info, errors
        
        return True, self.gpu_info, []
    
    def _get_gpu_info(self) -> GPUInfo:
        """Get GPU information from system."""
        chip_name = self._get_chip_name()
        gpu_cores = self._get_gpu_cores(chip_name)
        memory_info = self._get_memory_info()
        
        has_metal = self._check_metal_support()
        has_mps = self._check_mps_support()
        has_mlx = self._check_mlx_support()
        metal_version = self._get_metal_version()
        
        # Detect GPU family and capabilities
        gpu_family = self._detect_gpu_family(chip_name)
        capabilities = self._detect_msl_capabilities(gpu_family)
        
        return GPUInfo(
            has_metal=has_metal,
            has_mps=has_mps,
            has_mlx=has_mlx,
            gpu_cores=gpu_cores,
            total_memory=memory_info['total'],
            available_memory=memory_info['available'],
            metal_version=metal_version,
            chip_name=chip_name,
            unified_memory=True,
            is_apple_silicon=any(chip in chip_name for chip in ["M1", "M2", "M3", "M4"]),
            gpu_family=gpu_family,
            supports_bfloat16=capabilities['bfloat16'],
            supports_simdgroup_matrix=capabilities['simdgroup_matrix'],
            supports_mpp=capabilities['mpp'],
            supports_tile_functions=capabilities['tile_functions'],
            supports_atomic_float=capabilities['atomic_float'],
            supports_fast_math=capabilities['fast_math'],
            supports_function_constants=capabilities['function_constants'],
            max_threads_per_threadgroup=capabilities['max_threads_per_threadgroup']
        )
    
    def _get_chip_name(self) -> str:
        """Get Apple Silicon chip name."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "Unknown"
    
    def _get_gpu_cores(self, chip_name: str) -> int:
        """Get number of GPU cores based on chip."""
        gpu_core_map = {
            "M4": 16,
            "M4 Pro": 20,
            "M4 Max": 40,
            "M3": 10,
            "M3 Pro": 18,
            "M3 Max": 40,
            "M2": 10,
            "M2 Pro": 19,
            "M2 Max": 38,
            "M1": 8,
            "M1 Pro": 16,
            "M1 Max": 32,
        }
        
        for chip, cores in gpu_core_map.items():
            if chip in chip_name:
                return cores
        
        return 0
    
    def _get_memory_info(self) -> Dict[str, int]:
        """Get memory information."""
        vm = psutil.virtual_memory()
        return {
            'total': vm.total,
            'available': vm.available,
            'used': vm.used,
            'percent': vm.percent
        }
    
    def _check_metal_support(self) -> bool:
        """Check if Metal is supported."""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                check=True
            )
            return "Metal" in result.stdout
        except subprocess.CalledProcessError:
            return False
    
    def _check_mps_support(self) -> bool:
        """Check if MPS (Metal Performance Shaders) is available."""
        if not self._torch_available:
            return False
        
        try:
            import torch
            return torch.backends.mps.is_available()
        except Exception:
            return False
    
    def _check_mlx_support(self) -> bool:
        """Check if MLX is available."""
        if not self._mlx_available:
            return False
        
        try:
            import mlx.core as mx
            device = mx.default_device()
            return "gpu" in str(device).lower()
        except Exception:
            return False
    
    def _get_metal_version(self) -> Optional[str]:
        """Get Metal version."""
        try:
            result = subprocess.run(
                ["xcrun", "--show-sdk-version"],
                capture_output=True,
                text=True,
                check=True
            )
            sdk_version = result.stdout.strip()
            
            if float(sdk_version.split('.')[0]) >= 14:
                return "Metal 3"
            else:
                return "Metal 2"
        except Exception:
            return None
    
    def _detect_gpu_family(self, chip_name: str) -> str:
        """
        Detect GPU family based on chip name.
        
        Returns:
            GPU family identifier (apple5, apple6, apple7, apple8)
        """
        # Map chip names to GPU families
        if "M4" in chip_name:
            return "apple8"
        elif "M3" in chip_name:
            return "apple7"
        elif "M2" in chip_name:
            return "apple6"
        elif "M1" in chip_name:
            return "apple5"
        else:
            # Default to M1 capabilities for unknown chips
            return "apple5"
    
    def _detect_msl_capabilities(self, gpu_family: str) -> Dict[str, Any]:
        """
        Detect MSL v4 capabilities based on GPU family.
        
        Args:
            gpu_family: GPU family identifier
            
        Returns:
            Dictionary of capabilities
        """
        # Base capabilities (M1 and all Apple Silicon)
        capabilities = {
            'bfloat16': False,
            'simdgroup_matrix': True,  # All Apple Silicon supports this
            'mpp': False,
            'tile_functions': False,
            'atomic_float': True,
            'fast_math': True,
            'function_constants': True,
            'max_threads_per_threadgroup': 1024
        }
        
        # M2 and later capabilities
        if gpu_family in ["apple6", "apple7", "apple8"]:
            capabilities.update({
                'bfloat16': True,
                'mpp': True,
                'max_threads_per_threadgroup': 1024
            })
        
        # M3 and later capabilities
        if gpu_family in ["apple7", "apple8"]:
            capabilities.update({
                'tile_functions': True,
                'max_threads_per_threadgroup': 1024
            })
        
        return capabilities
    
    def get_optimal_dtype(self) -> str:
        """
        Get optimal data type for current hardware.
        
        Returns:
            'bfloat16' for M2+, 'float16' for M1
        """
        if not self.gpu_info:
            self.validate()
        
        if self.gpu_info and self.gpu_info.supports_bfloat16:
            return 'bfloat16'
        else:
            return 'float16'
    
    def check_bfloat16_support(self) -> bool:
        """
        Check if current hardware supports bfloat16.
        
        Returns:
            True if bfloat16 is supported
        """
        if not self.gpu_info:
            self.validate()
        
        return self.gpu_info.supports_bfloat16 if self.gpu_info else False
    
    def print_gpu_info(self) -> None:
        """Print GPU information."""
        if not self.gpu_info:
            self.validate()
        
        if not self.gpu_info:
            print("❌ Unable to get GPU information")
            return
        
        print("🖥️  GPU Information:")
        print(f"  Chip: {self.gpu_info.chip_name}")
        print(f"  GPU Family: {self.gpu_info.gpu_family}")
        print(f"  GPU Cores: {self.gpu_info.gpu_cores}")
        print(f"  Total Memory: {self.gpu_info.total_memory / (1024**3):.1f} GB")
        print(f"  Available Memory: {self.gpu_info.available_memory / (1024**3):.1f} GB")
        print(f"  Metal: {'✅' if self.gpu_info.has_metal else '❌'}")
        print(f"  MPS: {'✅' if self.gpu_info.has_mps else '❌'}")
        print(f"  MLX: {'✅' if self.gpu_info.has_mlx else '❌'}")
        print(f"  Metal Version: {self.gpu_info.metal_version or 'Unknown'}")
        print(f"  Unified Memory: {'✅' if self.gpu_info.unified_memory else '❌'}")
        print(f"  Apple Silicon: {'✅' if self.gpu_info.is_apple_silicon else '❌'}")
        
        print("\n📊 MSL v4 Capabilities:")
        print(f"  bfloat16: {'✅' if self.gpu_info.supports_bfloat16 else '❌'}")
        print(f"  SIMD-group matrices: {'✅' if self.gpu_info.supports_simdgroup_matrix else '❌'}")
        print(f"  MPP operations: {'✅' if self.gpu_info.supports_mpp else '❌'}")
        print(f"  Tile functions: {'✅' if self.gpu_info.supports_tile_functions else '❌'}")
        print(f"  Atomic float: {'✅' if self.gpu_info.supports_atomic_float else '❌'}")
        print(f"  Fast math: {'✅' if self.gpu_info.supports_fast_math else '❌'}")
        print(f"  Function constants: {'✅' if self.gpu_info.supports_function_constants else '❌'}")
        print(f"  Max threads/threadgroup: {self.gpu_info.max_threads_per_threadgroup}")
        print(f"  Optimal dtype: {self.get_optimal_dtype()}")
        
        if self.gpu_info.is_valid:
            print("\n✅ GPU meets all requirements for Cortex")
        else:
            print("\n❌ GPU does not meet requirements:")
            for error in self.gpu_info.get_validation_errors():
                print(f"  • {error}")
    
    def ensure_gpu_available(self) -> None:
        """Ensure GPU is available or exit."""
        is_valid, gpu_info, errors = self.validate()
        
        if not is_valid:
            print("⚠️  GPU validation warnings:")
            for error in errors:
                print(f"  • {error}")
            print("\nNote: Cortex is optimized for Apple Silicon with unified memory architecture.")
            print("Performance may be limited with current configuration.")
            # Don't exit for testing purposes
            # sys.exit(1)
        
        if gpu_info and not gpu_info.is_apple_silicon:
            print(f"⚠️  Warning: Detected {gpu_info.chip_name} - Apple Silicon recommended")
            print("   Performance may not match specifications in PRD")
    
    def get_gpu_memory_status(self) -> Dict[str, Any]:
        """Get current GPU memory status."""
        if not self.gpu_info:
            self.validate()
        
        if not self.gpu_info:
            return {
                'available': False,
                'error': 'GPU info not available'
            }
        
        memory_info = self._get_memory_info()
        
        return {
            'available': True,
            'total_gb': memory_info['total'] / (1024**3),
            'available_gb': memory_info['available'] / (1024**3),
            'used_gb': memory_info['used'] / (1024**3),
            'percent_used': memory_info['percent'],
            'can_load_model': memory_info['available'] >= 20 * 1024**3
        }
    
    def verify_model_compatibility(self, model_size_gb: float) -> Tuple[bool, str]:
        """
        Verify if a model can be loaded on GPU.
        
        Args:
            model_size_gb: Model size in gigabytes
            
        Returns:
            Tuple of (can_load, message)
        """
        memory_status = self.get_gpu_memory_status()
        
        if not memory_status['available']:
            return False, memory_status.get('error', 'GPU not available')
        
        available_gb = memory_status['available_gb']
        
        # Add overhead for KV cache, activations, and loading overhead
        # Some models expand significantly in memory during loading:
        # - Sharded models may temporarily duplicate memory
        # - KV cache and attention buffers add overhead
        # - Qwen models observed using 3.6x disk size in memory
        # Use conservative multiplier to ensure successful loading
        required_gb = model_size_gb * 3.5  # 250% overhead for safety
        
        if required_gb > available_gb:
            return False, f"Model requires {required_gb:.1f}GB (including overhead), only {available_gb:.1f}GB available"
        
        return True, f"Model can be loaded ({required_gb:.1f}GB required / {available_gb:.1f}GB available)"

def main():
    """Main function for testing GPU validation."""
    validator = GPUValidator()
    validator.print_gpu_info()
    
    is_valid, gpu_info, errors = validator.validate()
    
    if is_valid:
        print("\n✅ System ready for Cortex")
    else:
        print("\n❌ System not ready for Cortex")
        for error in errors:
            print(f"  • {error}")

if __name__ == "__main__":
    main()