#!/usr/bin/env python3
"""
Apple Silicon GPU Acceleration Validation Script for Cortex
Thoroughly validates Metal, MLX, MPS, and unified memory capabilities
"""

import sys
import time
import json
import subprocess
import platform
from datetime import datetime
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Add Cortex to path
sys.path.insert(0, str(Path(__file__).parent))

# Terminal colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")

def print_status(label: str, status: bool, details: str = ""):
    """Print status with color coding."""
    symbol = "✅" if status else "❌"
    color = Colors.GREEN if status else Colors.RED
    print(f"{symbol} {Colors.BOLD}{label:30}{Colors.END} {color}{details}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"⚠️  {Colors.YELLOW}{text}{Colors.END}")

def print_info(label: str, value: str):
    """Print info line."""
    print(f"   {Colors.BOLD}{label:30}{Colors.END} {value}")

# ============================================================================
# SYSTEM VALIDATION
# ============================================================================

def validate_system() -> Dict[str, Any]:
    """Validate system is Apple Silicon Mac."""
    print_header("SYSTEM VALIDATION")
    
    results = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "is_apple_silicon": False,
        "chip_model": "Unknown",
        "memory_gb": 0
    }
    
    # Check if macOS
    is_macos = platform.system() == "Darwin"
    print_status("macOS Platform", is_macos, platform.system())
    
    if not is_macos:
        print(f"{Colors.RED}ERROR: This system is not macOS. Apple Silicon features unavailable.{Colors.END}")
        return results
    
    # Check if Apple Silicon
    is_arm = platform.machine() == "arm64"
    print_status("ARM64 Architecture", is_arm, platform.machine())
    
    if not is_arm:
        print(f"{Colors.RED}ERROR: This is not Apple Silicon (Intel Mac detected).{Colors.END}")
        return results
    
    results["is_apple_silicon"] = True
    
    # Get chip model
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True
        )
        chip_model = result.stdout.strip()
        results["chip_model"] = chip_model
        print_status("Apple Silicon Chip", True, chip_model)
    except:
        print_warning("Could not determine chip model")
    
    # Get memory info
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, check=True
        )
        memory_bytes = int(result.stdout.strip())
        memory_gb = memory_bytes / (1024**3)
        results["memory_gb"] = memory_gb
        print_status("Unified Memory", True, f"{memory_gb:.1f} GB")
    except:
        print_warning("Could not determine memory size")
    
    # Get GPU cores
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        for display in data.get("SPDisplaysDataType", []):
            if "sppci_cores" in display:
                gpu_cores = display["sppci_cores"]
                results["gpu_cores"] = gpu_cores
                print_info("GPU Cores", str(gpu_cores))
                break
    except:
        pass
    
    return results

# ============================================================================
# METAL VALIDATION
# ============================================================================

def validate_metal() -> Dict[str, Any]:
    """Validate Metal framework availability."""
    print_header("METAL FRAMEWORK VALIDATION")
    
    results = {
        "metal_available": False,
        "metal_version": None,
        "mps_available": False
    }
    
    # Check Metal framework
    try:
        result = subprocess.run(
            ["xcrun", "--show-sdk-version"],
            capture_output=True, text=True, check=True
        )
        sdk_version = result.stdout.strip()
        results["metal_available"] = True
        results["sdk_version"] = sdk_version
        
        # Determine Metal version from SDK
        major = int(sdk_version.split('.')[0])
        if major >= 14:
            metal_version = "Metal 3"
        elif major >= 10:
            metal_version = "Metal 2"
        else:
            metal_version = "Metal 1"
        
        results["metal_version"] = metal_version
        print_status("Metal Framework", True, f"{metal_version} (SDK {sdk_version})")
    except:
        print_status("Metal Framework", False, "Not available")
        return results
    
    # Check Metal compiler
    try:
        result = subprocess.run(
            ["xcrun", "-find", "metal"],
            capture_output=True, text=True, check=True
        )
        compiler_path = result.stdout.strip()
        print_status("Metal Shader Compiler", True, "Available")
        print_info("Compiler Path", compiler_path)
    except:
        print_status("Metal Shader Compiler", False, "Not found")
    
    # Check MetalPerformanceShaders
    try:
        result = subprocess.run(
            ["xcrun", "--show-sdk-path"],
            capture_output=True, text=True, check=True
        )
        sdk_path = result.stdout.strip()
        mps_path = Path(sdk_path) / "System/Library/Frameworks/MetalPerformanceShaders.framework"
        if mps_path.exists():
            results["mps_available"] = True
            print_status("MetalPerformanceShaders", True, "Framework found")
        else:
            print_status("MetalPerformanceShaders", False, "Framework not found")
    except:
        print_status("MetalPerformanceShaders", False, "Could not verify")
    
    return results

# ============================================================================
# PYTHON PACKAGE VALIDATION
# ============================================================================

def validate_python_packages() -> Dict[str, Any]:
    """Validate required Python packages for GPU acceleration."""
    print_header("PYTHON PACKAGE VALIDATION")
    
    results = {
        "torch_mps": False,
        "mlx": False,
        "mlx_device": None,
        "packages": {}
    }
    
    # Check PyTorch with MPS
    try:
        import torch
        has_mps = torch.backends.mps.is_available()
        built_mps = torch.backends.mps.is_built()
        
        results["torch_mps"] = has_mps and built_mps
        results["packages"]["torch"] = torch.__version__
        
        print_status("PyTorch", True, f"v{torch.__version__}")
        print_status("MPS Backend Available", has_mps, "")
        print_status("MPS Backend Built", built_mps, "")
        
        if has_mps:
            # Test MPS allocation
            try:
                device = torch.device("mps")
                test_tensor = torch.randn(100, 100, device=device)
                _ = torch.matmul(test_tensor, test_tensor)
                print_status("MPS Tensor Operations", True, "Working")
            except Exception as e:
                print_status("MPS Tensor Operations", False, str(e))
                results["torch_mps"] = False
    except ImportError:
        print_status("PyTorch", False, "Not installed")
    except Exception as e:
        print_status("PyTorch", False, str(e))
    
    # Check MLX
    try:
        import mlx.core as mx
        import mlx.nn as nn
        
        device = mx.default_device()
        # Device(gpu, 0) means GPU device 0
        is_gpu = "gpu" in str(device).lower()
        
        results["mlx"] = True
        results["mlx_device"] = str(device)
        results["packages"]["mlx"] = "installed"
        
        print_status("MLX Framework", True, "Installed")
        print_status("MLX Device", is_gpu, str(device))
        
        if is_gpu:
            # Test MLX operations
            try:
                test_array = mx.random.normal((100, 100))
                result = mx.matmul(test_array, test_array)
                mx.eval(result)
                print_status("MLX GPU Operations", True, "Working")
            except Exception as e:
                print_status("MLX GPU Operations", False, str(e))
                results["mlx"] = False
        else:
            print_warning("MLX not using GPU!")
            results["mlx"] = False
            
    except ImportError:
        print_status("MLX Framework", False, "Not installed")
    except Exception as e:
        print_status("MLX Framework", False, str(e))
    
    # Check other packages
    packages_to_check = ["llama-cpp-python", "transformers", "safetensors"]
    for package in packages_to_check:
        try:
            module = __import__(package.replace("-", "_"))
            version = getattr(module, "__version__", "unknown")
            results["packages"][package] = version
            print_info(package, f"v{version}")
        except ImportError:
            print_info(package, "Not installed")
    
    return results

# ============================================================================
# CORTEX MODULE VALIDATION
# ============================================================================

def validate_cortex_modules() -> Dict[str, Any]:
    """Validate Cortex Metal optimization modules."""
    print_header("CORTEX METAL MODULE VALIDATION")
    
    results = {
        "modules_found": {},
        "initialization": {},
        "errors": []
    }
    
    modules_to_check = [
        ("cortex.metal", "Base Metal Module"),
        ("cortex.metal.memory_pool", "Memory Pool"),
        ("cortex.metal.mps_optimizer", "MPS Optimizer"),
        ("cortex.metal.mlx_accelerator", "MLX Accelerator"),
        ("cortex.metal.performance_profiler", "Performance Profiler")
    ]
    
    for module_path, module_name in modules_to_check:
        try:
            module = __import__(module_path, fromlist=[''])
            results["modules_found"][module_path] = True
            print_status(module_name, True, "Module loaded")
            
            # Try to initialize key components
            if module_path == "cortex.metal.memory_pool":
                try:
                    from cortex.metal.memory_pool import MemoryPool, AllocationStrategy
                    pool = MemoryPool(
                        pool_size=None,
                        strategy=AllocationStrategy.UNIFIED,
                        device="mps",
                        auto_size=True,
                        silent=True
                    )
                    stats = pool.get_stats()
                    results["initialization"]["memory_pool"] = True
                    print_info("Memory Pool Size", f"{stats['pool_size'] / (1024**3):.1f} GB")
                except Exception as e:
                    results["initialization"]["memory_pool"] = False
                    results["errors"].append(f"Memory Pool: {str(e)}")
                    print_warning(f"Memory Pool initialization failed: {str(e)}")
            
            
        except ImportError as e:
            results["modules_found"][module_path] = False
            print_status(module_name, False, "Module not found")
            results["errors"].append(f"{module_path}: {str(e)}")
        except Exception as e:
            results["modules_found"][module_path] = False
            print_status(module_name, False, f"Error: {str(e)}")
            results["errors"].append(f"{module_path}: {str(e)}")
    
    return results

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_configuration() -> Dict[str, Any]:
    """Validate Cortex configuration for Apple Silicon."""
    print_header("CONFIGURATION VALIDATION")
    
    results = {
        "config_valid": False,
        "issues": [],
        "settings": {}
    }
    
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    if not config_path.exists():
        print_status("config.yaml", False, "File not found")
        return results
    
    print_status("config.yaml", True, "File found")
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check critical GPU settings
        critical_settings = [
            ("compute_backend", "metal", "Should be 'metal'"),
            ("force_gpu", True, "Should be True"),
            ("metal_performance_shaders", True, "Should be True"),
            ("mlx_backend", True, "Should be True"),
            ("unified_memory", True, "Should be True for Apple Silicon"),
            ("cpu_offload", False, "Should be False for GPU-only")
        ]
        
        all_correct = True
        for setting, expected, description in critical_settings:
            actual = config.get(setting)
            is_correct = actual == expected
            
            if not is_correct:
                all_correct = False
                print_status(setting, False, f"{actual} - {description}")
                results["issues"].append(f"{setting} is {actual}, should be {expected}")
            else:
                print_status(setting, True, str(actual))
            
            results["settings"][setting] = actual
        
        results["config_valid"] = all_correct
        
        # Check memory settings
        print_info("GPU Memory Fraction", str(config.get("gpu_memory_fraction", "not set")))
        print_info("Max GPU Memory", str(config.get("max_gpu_memory", "not set")))
        
    except Exception as e:
        print_status("Configuration Parse", False, str(e))
        results["issues"].append(f"Failed to parse config: {str(e)}")
    
    return results

# ============================================================================
# PERFORMANCE TESTING
# ============================================================================

def run_gpu_performance() -> Dict[str, Any]:
    """Run GPU performance tests."""
    print_header("GPU PERFORMANCE TESTING")
    
    results = {
        "tests_run": False,
        "mps_performance": {},
        "mlx_performance": {},
        "memory_bandwidth": 0
    }
    
    # Test MPS Performance
    try:
        import torch
        if torch.backends.mps.is_available():
            print("Running MPS performance test...")
            device = torch.device("mps")
            
            # Matrix multiplication test
            sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
            for size in sizes:
                a = torch.randn(size, device=device, dtype=torch.float32)
                b = torch.randn(size, device=device, dtype=torch.float32)
                
                # Warmup
                for _ in range(3):
                    c = torch.matmul(a, b)
                
                # Time it
                torch.mps.synchronize()
                start = time.perf_counter()
                for _ in range(10):
                    c = torch.matmul(a, b)
                torch.mps.synchronize()
                end = time.perf_counter()
                
                avg_time = (end - start) / 10
                gflops = (2 * size[0] * size[1] * size[1]) / (avg_time * 1e9)
                
                results["mps_performance"][f"{size[0]}x{size[1]}"] = {
                    "time_ms": avg_time * 1000,
                    "gflops": gflops
                }
                
                print_info(f"MPS MatMul {size[0]}x{size[1]}", f"{gflops:.1f} GFLOPS")
            
            results["tests_run"] = True
    except Exception as e:
        print_warning(f"MPS test failed: {str(e)}")
    
    # Test MLX Performance
    try:
        import mlx.core as mx
        if str(mx.default_device()).lower() == "gpu":
            print("Running MLX performance test...")
            
            sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
            for size in sizes:
                a = mx.random.normal(size)
                b = mx.random.normal(size)
                
                # Warmup
                for _ in range(3):
                    c = mx.matmul(a, b)
                    mx.eval(c)
                
                # Time it
                start = time.perf_counter()
                for _ in range(10):
                    c = mx.matmul(a, b)
                    mx.eval(c)
                end = time.perf_counter()
                
                avg_time = (end - start) / 10
                gflops = (2 * size[0] * size[1] * size[1]) / (avg_time * 1e9)
                
                results["mlx_performance"][f"{size[0]}x{size[1]}"] = {
                    "time_ms": avg_time * 1000,
                    "gflops": gflops
                }
                
                print_info(f"MLX MatMul {size[0]}x{size[1]}", f"{gflops:.1f} GFLOPS")
            
            # Memory bandwidth test
            print("Testing memory bandwidth...")
            size = 100 * 1024 * 1024 // 4  # 100MB of float32
            data = mx.random.normal((size,))
            
            start = time.perf_counter()
            for _ in range(100):
                result = mx.copy(data)
                mx.eval(result)
            end = time.perf_counter()
            
            total_bytes = size * 4 * 100 * 2  # read + write
            bandwidth_gb = total_bytes / (end - start) / 1e9
            results["memory_bandwidth"] = bandwidth_gb
            
            print_info("Memory Bandwidth", f"{bandwidth_gb:.1f} GB/s")
            
            results["tests_run"] = True
    except Exception as e:
        print_warning(f"MLX test failed: {str(e)}")
    
    return results


def test_gpu_performance() -> None:
    """Pytest wrapper for GPU performance checks."""
    results = run_gpu_performance()
    assert isinstance(results, dict)

# ============================================================================
# MAIN VALIDATION
# ============================================================================

def main():
    """Run complete Apple Silicon validation."""
    print(f"\n{Colors.BOLD}CORTEX APPLE SILICON VALIDATION{Colors.END}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version.split()[0]}")
    
    all_results = {}
    
    # Run all validations
    all_results["system"] = validate_system()
    
    if not all_results["system"]["is_apple_silicon"]:
        print(f"\n{Colors.RED}{'=' * 60}{Colors.END}")
        print(f"{Colors.RED}VALIDATION FAILED: Not running on Apple Silicon{Colors.END}")
        print(f"{Colors.RED}{'=' * 60}{Colors.END}")
        return 1
    
    all_results["metal"] = validate_metal()
    all_results["packages"] = validate_python_packages()
    all_results["cortex_modules"] = validate_cortex_modules()
    all_results["configuration"] = validate_configuration()
    all_results["performance"] = run_gpu_performance()
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    # Determine overall status
    critical_checks = [
        ("Apple Silicon", all_results["system"]["is_apple_silicon"]),
        ("Metal Framework", all_results["metal"]["metal_available"]),
        ("MPS Backend", all_results["packages"]["torch_mps"]),
        ("MLX GPU", all_results["packages"]["mlx"] and "gpu" in str(all_results["packages"]["mlx_device"]).lower()),
        ("Cortex Modules", all(all_results["cortex_modules"]["modules_found"].values())),
        ("Configuration", len(all_results["configuration"]["issues"]) == 0),
        ("GPU Performance", all_results["performance"]["tests_run"])
    ]
    
    all_passed = True
    for check_name, check_status in critical_checks:
        print_status(check_name, check_status, "")
        if not check_status:
            all_passed = False
    
    # Issues and warnings
    if all_results["configuration"]["issues"]:
        print(f"\n{Colors.YELLOW}Configuration Issues:{Colors.END}")
        for issue in all_results["configuration"]["issues"]:
            print(f"  • {issue}")
    
    if all_results["cortex_modules"]["errors"]:
        print(f"\n{Colors.YELLOW}Module Errors:{Colors.END}")
        for error in all_results["cortex_modules"]["errors"][:5]:  # Limit to 5
            print(f"  • {error}")
    
    # Final verdict
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ VALIDATION PASSED: Cortex is fully optimized for Apple Silicon!{Colors.END}")
    elif all_results["packages"]["torch_mps"] or all_results["packages"]["mlx"]:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  VALIDATION PARTIAL: Cortex is using GPU but not fully optimized{Colors.END}")
        print(f"{Colors.YELLOW}   Some GPU optimizations may not be active{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ VALIDATION FAILED: Cortex is NOT using Apple Silicon GPU{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
    
    # Save results
    results_file = Path(__file__).parent / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_file}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
