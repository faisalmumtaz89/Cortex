# Metal Acceleration Module

This directory contains Python wrappers for GPU acceleration on Apple Silicon via MLX and PyTorch MPS frameworks. There are no custom Metal kernels or Metal Shading Language (MSL) code in this module -- all GPU computation is delegated to MLX and PyTorch's built-in Metal implementations.

## Files

```
metal/
├── README.md                  # This file
├── __init__.py                # Metal capabilities detection and framework imports
├── mlx_accelerator.py         # MLX framework integration (primary backend)
├── mlx_compat.py              # MLX/MLX-LM compatibility patches
├── mlx_converter.py           # Model conversion to MLX format with quantization
├── mps_optimizer.py           # PyTorch MPS backend optimization
├── optimizer.py               # Unified optimizer interface (MLX + MPS abstraction)
├── memory_pool.py             # GPU memory pool management with allocation strategies
├── gpu_validator.py           # Hardware capability detection (Metal GPU family)
└── performance_profiler.py    # Runtime performance monitoring (timing, GPU utilization, memory)
```

## Architecture

### MLX Accelerator (`mlx_accelerator.py`)
- Primary inference backend using Apple's MLX framework
- Configures JIT compilation (`mx.compile`), operation fusion, and lazy evaluation
- Implements rotating KV cache for long context windows
- Provides `optimized_attention()` using MLX's built-in `mx.matmul` and `mx.softmax`
- Weight matrix padding for alignment optimization
- Quantization support: 4-bit, 8-bit, mixed-precision (5-bit available via converter)

### MLX Converter (`mlx_converter.py`)
- Converts HuggingFace models to MLX format
- Quantization recipes: SPEED_4BIT, BALANCED_5BIT, QUALITY_8BIT, MIXED_PRECISION
- Group-wise quantization with configurable group size
- Conversion caching with metadata

### MPS Optimizer (`mps_optimizer.py`)
- PyTorch Metal Performance Shaders backend wrapper
- FP16 (half-precision) optimization
- Channels-last tensor format for improved cache behavior
- Graph optimization and operation fusion at the PyTorch level

### Memory Pool (`memory_pool.py`)
- Pre-allocated GPU memory buffers (MPS and MLX)
- Allocation strategies: BEST_FIT, FIRST_FIT, UNIFIED, DEDICATED, ZERO_COPY
- Automatic pool sizing: 60% of available memory, capped at 75% of total, hard cap at 20GB
- Block-based allocation with defragmentation support

### GPU Validator (`gpu_validator.py`)
- Detects Metal GPU support via system_profiler
- Identifies Apple Silicon GPU family (M1/M2/M3/M4)
- Returns GPU capabilities for downstream use
- Note: MPS and MLX software availability checks are in `__init__.py` and `optimizer.py`

### Performance Profiler (`performance_profiler.py`)
- Operation timing and GPU utilization monitoring
- System memory usage tracking via psutil
- FLOPS estimation from timed operations

### Unified Optimizer (`optimizer.py`)
- Abstraction layer over MLX and MPS backends
- Automatic backend selection based on framework availability
- `InferenceSession` for managing generation lifecycle
