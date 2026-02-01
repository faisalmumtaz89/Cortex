# GPU Validation

## Overview

Cortex requires Apple Silicon with Metal support for GPU-accelerated inference. The GPU validation system ensures your hardware meets the minimum requirements and provides detailed information about available GPU capabilities.

## Requirements

### Minimum Hardware Requirements

- **Platform**: macOS on Apple Silicon (M1, M2, M3, M4 series)
- **Architecture**: ARM64
- **Memory**: 8GB unified memory recommended (validation minimum: 4GB)
- **GPU Cores**: Minimum 8 cores
- **macOS Version**: 13.3+ recommended (for MLX/Metal 3 support)

### Software Requirements

- **Metal**: Metal API support (included in macOS)
- **Metal Performance Shaders (MPS)**: For PyTorch acceleration
- **MLX Framework**: Apple's ML framework for Apple Silicon
- **PyTorch**: With MPS backend support

## GPUInfo Class

**GPU information and capabilities data structure:**

```python
@dataclass
class GPUInfo:
    """GPU information and capabilities."""
    has_metal: bool            # Metal API available
    has_mps: bool              # Metal Performance Shaders available
    has_mlx: bool              # MLX framework available
    gpu_cores: int             # Number of GPU cores
    total_memory: int          # Total system memory (bytes)
    available_memory: int      # Available memory (bytes)
    metal_version: Optional[str]  # Metal version string
    chip_name: str             # Apple Silicon chip name (e.g., "M2")
    unified_memory: bool       # Unified memory architecture
    is_apple_silicon: bool     # Apple Silicon chip detected

    # MSL v4 capabilities
    gpu_family: str            # apple5 (M1), apple6 (M2), apple7 (M3), apple8 (M4)
    supports_bfloat16: bool
    supports_simdgroup_matrix: bool
    supports_mpp: bool
    supports_tile_functions: bool
    supports_atomic_float: bool
    supports_fast_math: bool
    supports_function_constants: bool
    max_threads_per_threadgroup: int
```

### Validation Properties

```python
@property
def is_valid(self) -> bool:
    """Check if GPU meets requirements."""
    min_memory = 4 * 1024 * 1024 * 1024  # 4GB minimum
    min_cores = 8  # M1 and above have at least 8 cores
    
    return (
        self.has_metal and
        self.has_mps and
        self.has_mlx and
        self.gpu_cores >= min_cores and
        self.available_memory >= min_memory
    )
```

### Error Reporting

```python
def get_validation_errors(self) -> list[str]:
    """Get list of validation errors."""
    errors = []
    
    if not self.has_metal:
        errors.append("Metal support not available")
    if not self.has_mps:
        errors.append("Metal Performance Shaders (MPS) not available")
    if not self.has_mlx:
        errors.append("MLX framework not available")
    if self.gpu_cores < min_cores:
        errors.append(f"Insufficient GPU cores: {self.gpu_cores}")
    if self.available_memory < min_memory:
        errors.append(f"Insufficient GPU memory: {memory_gb:.1f}GB")
    
    return errors
```

## GPUValidator Class

**Main validation and detection class:**

```python
class GPUValidator:
    """Validate GPU capabilities for Cortex."""
    
    def __init__(self, config=None):
        """Initialize GPU validator."""
        self.config = config
        self.gpu_info: Optional[GPUInfo] = None
        self._torch_available = False
        self._mlx_available = False
        self._validate_imports()
```

### Core Validation Method

```python
def validate(self) -> Tuple[bool, Optional[GPUInfo], list[str]]:
    """
    Validate GPU support.
    
    Returns:
        Tuple of (is_valid, gpu_info, errors)
    """
    errors = []
    
    # Check platform
    if platform.system().lower() != "darwin":
        errors.append(f"macOS required, found {platform.system()}")
        return False, None, errors
    
    # Check architecture
    if platform.machine() != "arm64":
        errors.append(f"ARM64 required, found {platform.machine()}")
        return False, None, errors
    
    # Detect GPU capabilities
    gpu_info = self._get_gpu_info()
    
    # Validate requirements
    if not gpu_info.is_valid:
        errors.extend(gpu_info.get_validation_errors())
        return False, gpu_info, errors
    
    return True, gpu_info, []
```

### GPU Detection

```python
def _get_gpu_info(self) -> GPUInfo:
    """Detect GPU capabilities."""
    has_metal = self._check_metal_support()
    has_mps = self._check_mps_support()
    has_mlx = self._mlx_available
    
    # Get chip information
    chip_name = self._get_chip_name()
    gpu_cores = self._get_gpu_cores(chip_name)
    
    # Get memory information
    total_memory = psutil.virtual_memory().total
    available_memory = psutil.virtual_memory().available
    
    return GPUInfo(
        has_metal=has_metal,
        has_mps=has_mps,
        has_mlx=has_mlx,
        gpu_cores=gpu_cores,
        total_memory=total_memory,
        available_memory=available_memory,
        metal_version=self._get_metal_version(),
        chip_name=chip_name,
        unified_memory=True,  # Always true for Apple Silicon
        is_apple_silicon=True
    )
```

### Metal Support Detection

```python
def _check_metal_support(self) -> bool:
    """Check if Metal is available."""
    try:
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'],
            capture_output=True,
            text=True,
            check=False
        )
        return 'Metal' in result.stdout
    except Exception:
        return False
```

### MPS Support Detection

```python
def _check_mps_support(self) -> bool:
    """Check if Metal Performance Shaders is available."""
    if not self._torch_available:
        return False
    
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False
```

### Chip Detection

```python
def _get_chip_name(self) -> str:
    """Get Apple Silicon chip name."""
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True,
            text=True,
            check=False
        )
        brand_string = result.stdout.strip()
        
        # Extract chip name (M1, M2, etc.)
        if 'Apple M' in brand_string:
            parts = brand_string.split()
            for part in parts:
                if part.startswith('M') and part[1].isdigit():
                    return part
        
        return brand_string
    except Exception:
        return "Unknown"
```

### GPU Core Count Mapping

```python
def _get_gpu_cores(self, chip_name: str) -> int:
    """Get GPU core count for Apple Silicon chip."""
    # GPU core counts for different chips
    gpu_core_map = {
        'M4': 16,
        'M4 Pro': 20,
        'M4 Max': 40,
        'M3': 10,
        'M3 Pro': 18,
        'M3 Max': 40,
        'M2': 10,
        'M2 Pro': 19,
        'M2 Max': 38,
        'M1': 8,
        'M1 Pro': 16,
        'M1 Max': 32,
    }
    
    # Find matching chip
    for chip, cores in gpu_core_map.items():
        if chip in chip_name:
            return cores

    # Default fallback
    return 0
```

## Usage Examples

### Basic Validation

```python
from cortex.gpu_validator import GPUValidator

# Create validator
validator = GPUValidator()

# Run validation
is_valid, gpu_info, errors = validator.validate()

if is_valid:
    print(f"OK - GPU validated: {gpu_info.chip_name}")
    print(f"  GPU Cores: {gpu_info.gpu_cores}")
    print(f"  Memory: {gpu_info.total_memory / (1024**3):.1f}GB")
else:
    print("GPU validation failed:")
    for error in errors:
        print(f"  - {error}")
```

### Detailed Information

```python
# Get detailed GPU information
is_valid, gpu_info, _ = validator.validate()

if gpu_info:
    print(f"Chip: {gpu_info.chip_name}")
    print(f"GPU Cores: {gpu_info.gpu_cores}")
    print(f"Total Memory: {gpu_info.total_memory / (1024**3):.1f}GB")
    print(f"Available Memory: {gpu_info.available_memory / (1024**3):.1f}GB")
    print(f"Metal: {'Yes' if gpu_info.has_metal else 'No'}")
    print(f"MPS: {'Yes' if gpu_info.has_mps else 'No'}")
    print(f"MLX: {'Yes' if gpu_info.has_mlx else 'No'}")
    print(f"Metal Version: {gpu_info.metal_version or 'Unknown'}")
```

### Integration with Config

```python
from cortex.config import Config
from cortex.gpu_validator import GPUValidator

# Load configuration
config = Config()

# Validate with config
validator = GPUValidator(config)
is_valid, gpu_info, errors = validator.validate()

# Check against config requirements
if config.gpu.force_gpu and not is_valid:
    print("Error: GPU required but validation failed")
    sys.exit(1)
```

## Validation Process

### Startup Validation

When Cortex starts, it performs the following validation steps:

1. **Platform Check**: Verify macOS on ARM64
2. **Import Check**: Validate PyTorch and MLX availability
3. **Metal Detection**: Check Metal API support
4. **MPS Detection**: Verify MPS backend for PyTorch
5. **Chip Detection**: Identify Apple Silicon model
6. **Memory Check**: Ensure sufficient unified memory
7. **Core Count**: Verify minimum GPU cores

### Validation Errors

Common validation errors and solutions:

| Error | Solution |
|-------|----------|
| "macOS required" | Cortex only runs on macOS |
| "ARM64 architecture required" | Requires Apple Silicon Mac |
| "Metal support not available" | Update macOS to 13.3+ |
| "MPS not available" | Install PyTorch with MPS support |
| "MLX framework not available" | Run `pip install "mlx>=0.30.4" "mlx-lm>=0.30.5"` |
| "Insufficient GPU memory" | Close other applications or upgrade hardware |
| "Insufficient GPU cores" | Requires M1 or newer chip |

## Memory Requirements

### Model Size Guidelines

| Model Size | Minimum Memory | Recommended Memory |
|------------|----------------|-------------------|
| 7B parameters | 8GB | 16GB |
| 13B parameters | 16GB | 32GB |
| 30B parameters | 32GB | 64GB |
| 70B parameters | 64GB | 128GB |

### Memory Calculation

```python
def estimate_memory_requirement(model_params_billions: float, 
                               quantization: str = "Q4_K_M") -> float:
    """Estimate memory requirement for model."""
    # Bits per parameter for different quantizations
    bits_per_param = {
        "FP32": 32,
        "FP16": 16,
        "Q8_0": 8,
        "Q6_K": 6,
        "Q5_K_M": 5,
        "Q4_K_M": 4,
    }
    
    bits = bits_per_param.get(quantization, 4)
    bytes_per_param = bits / 8
    
    # Calculate base model size
    model_size_gb = model_params_billions * bytes_per_param
    
    # Add overhead for KV cache and activations (20%)
    total_gb = model_size_gb * 1.2
    
    return total_gb
```

## Performance Optimization

## Troubleshooting

### Common Issues

**"MLX not found"**
```bash
pip install "mlx>=0.30.4" "mlx-lm>=0.30.5"
```

**"PyTorch MPS not available"**
```bash
pip install torch torchvision torchaudio
```

**"Metal version too old"**
- Update macOS to 14.0 or later
- Check system requirements

**"Insufficient memory"**
- Close memory-intensive applications
- Use quantized models (Q4_K_M)
- Reduce batch size in configuration

### Debug Mode

Enable debug output for detailed validation:

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run validation with debug output
validator = GPUValidator()
is_valid, gpu_info, errors = validator.validate()
```

## Best Practices

1. **Always validate on startup** to ensure hardware compatibility
2. **Check memory availability** before loading models
3. **Monitor GPU utilization** during inference
4. **Use appropriate quantization** for your hardware
5. **Enable all optimizations** (MPS, MLX, Flash Attention)
6. **Keep macOS updated** for latest Metal features
7. **Profile performance** to identify bottlenecks
