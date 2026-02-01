"""GPU validation and capability detection for Metal."""

import subprocess
import platform
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPUInfo:
    """GPU information and capabilities."""
    gpu_family: str = "unknown"  # apple5 (M1), apple6 (M2), apple7 (M3), apple8 (M4)
    supports_bfloat16: bool = False
    supports_simdgroup_matrix: bool = False
    supports_tile_functions: bool = False
    supports_mpp: bool = False
    is_apple_silicon: bool = False
    metal_version: str = "3.0"

class GPUValidator:
    """Validates GPU capabilities for Metal optimization."""
    
    def __init__(self):
        """Initialize GPU validator."""
        self.gpu_info = None
        self.validation_passed = False
    
    def validate(self) -> bool:
        """
        Validate GPU and detect capabilities.
        
        Returns:
            True if GPU is validated and ready
        """
        self.gpu_info = self._detect_gpu()
        self.validation_passed = self.gpu_info is not None
        return self.validation_passed
    
    def _detect_gpu(self) -> Optional[GPUInfo]:
        """
        Detect GPU model and capabilities.
        
        Returns:
            GPUInfo object with detected capabilities
        """
        info = GPUInfo()
        
        if platform.system() != "Darwin":
            return None
        
        try:
            # Use system_profiler to detect GPU
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                output = result.stdout.lower()
                
                # Detect Apple Silicon
                if "apple m" in output or "apple silicon" in output:
                    info.is_apple_silicon = True
                    
                    # Detect specific chip
                    if "m4" in output:
                        info.gpu_family = "apple8"
                        info.supports_bfloat16 = True
                        info.supports_tile_functions = True
                        info.metal_version = "3.1"
                    elif "m3" in output:
                        info.gpu_family = "apple7"
                        info.supports_bfloat16 = True
                        info.supports_tile_functions = True
                        info.metal_version = "3.1"
                    elif "m2" in output:
                        info.gpu_family = "apple6"
                        info.supports_bfloat16 = True
                        info.metal_version = "3.0"
                    elif "m1" in output:
                        info.gpu_family = "apple5"
                        info.supports_bfloat16 = False
                        info.metal_version = "3.0"
                    
                    # All Apple Silicon supports SIMD operations
                    info.supports_simdgroup_matrix = False  # Not in public API
                    info.supports_mpp = True
                
                return info
                
        except (subprocess.TimeoutExpired, Exception) as e:
            # Fallback detection
            try:
                # Try sysctl for chip detection
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True
                )
                if "Apple" in result.stdout:
                    info.is_apple_silicon = True
                    info.gpu_family = "apple5"  # Conservative default
                    return info
            except:
                pass
        
        return info if info.is_apple_silicon else None
    
    def check_bfloat16_support(self) -> bool:
        """
        Check if current GPU supports bfloat16.
        
        Returns:
            True if bfloat16 is supported
        """
        if not self.gpu_info:
            self.validate()
        
        return self.gpu_info.supports_bfloat16 if self.gpu_info else False
    
    def get_gpu_family(self) -> str:
        """
        Get GPU family identifier.
        
        Returns:
            GPU family string (apple5, apple6, etc.)
        """
        if not self.gpu_info:
            self.validate()
        
        return self.gpu_info.gpu_family if self.gpu_info else "unknown"
    
    def get_metal_version(self) -> str:
        """
        Get recommended Metal version for this GPU.
        
        Returns:
            Metal version string
        """
        if not self.gpu_info:
            self.validate()
        
        return self.gpu_info.metal_version if self.gpu_info else "3.0"
    
    def get_capabilities_summary(self) -> dict:
        """
        Get summary of GPU capabilities.
        
        Returns:
            Dictionary with capability flags
        """
        if not self.gpu_info:
            self.validate()
        
        if self.gpu_info:
            return {
                "gpu_family": self.gpu_info.gpu_family,
                "is_apple_silicon": self.gpu_info.is_apple_silicon,
                "supports_bfloat16": self.gpu_info.supports_bfloat16,
                "metal_version": self.gpu_info.metal_version,
                "validation_passed": self.validation_passed
            }
        
        return {
            "gpu_family": "unknown",
            "is_apple_silicon": False,
            "supports_bfloat16": False,
            "metal_version": "3.0",
            "validation_passed": False
        }

# Convenience function for quick GPU validation
def validate_gpu() -> bool:
    """Quick validation function."""
    validator = GPUValidator()
    return validator.validate()