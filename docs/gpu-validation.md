# GPU Validation

## Overview

Cortex requires Apple Silicon with Metal support for local inference. The validator (`cortex/gpu_validator.py`) checks the hardware and the MLX framework, and reports detailed GPU capabilities used by `/gpu` and `/status`.

## Requirements

- **Platform**: macOS on Apple Silicon (M1, M2, M3, M4 series), ARM64
- **Metal**: Metal API support (included in macOS; 13.3+ recommended)
- **MLX**: the `mlx` package must be importable
- **Memory**: at least 4GB available unified memory (8GB+ recommended)
- **GPU cores**: at least 8 (all M1-and-newer chips qualify)

There is no PyTorch/MPS dependency; MLX and llama.cpp are the only local backends.

## What Gets Checked

`GPUValidator.validate()` returns `(is_valid, gpu_info, errors)`:

1. **Platform**: `darwin` + `arm64`
2. **Metal**: detected via `system_profiler SPDisplaysDataType`
3. **MLX**: `mlx.core` importable
4. **Chip**: name from `sysctl machdep.cpu.brand_string`, mapped to a known GPU core count
5. **Memory**: total/available unified memory via `psutil`

`GPUInfo` also carries capability flags (GPU family, bfloat16, simdgroup matrix, tile functions, etc.) used by the Metal layer.

## Behavior on Failure

- **Worker mode** (`--worker-stdio`) and **headless mode** (`-p`) run validation non-strictly: a failure logs a warning and disables local inference, but cloud models still work.
- Loading a local model additionally checks that the model size fits available memory (`verify_model_compatibility`).

## Common Validation Errors

| Error | Solution |
|-------|----------|
| "macOS required" | Cortex only runs on macOS |
| "ARM64 architecture required" | Requires an Apple Silicon Mac |
| "Metal support not available" | Update macOS (13.3+) |
| "MLX framework not available" | `pip install "mlx>=0.30.4" "mlx-lm>=0.30.5"` |
| "Insufficient GPU memory" | Close other applications, or use a smaller model |
| "Insufficient GPU cores" | Requires M1 or newer |

## Memory Guidelines

Rough sizing for local models (4-bit quantized weights plus ~20% overhead for KV cache and activations):

| Model size | Minimum memory | Recommended |
|------------|----------------|-------------|
| 3B | 8GB | 16GB |
| 7B | 8GB | 16GB |
| 13B | 16GB | 32GB |
| 30B | 32GB | 64GB |
| 70B | 64GB | 128GB |

Check actual usage with `/gpu` after loading a model.
