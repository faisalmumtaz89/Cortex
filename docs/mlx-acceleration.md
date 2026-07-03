# MLX Acceleration

## Overview

MLX (Apple's ML framework) is Cortex's primary local inference backend, giving native Metal GPU execution on Apple Silicon. The integration lives in `cortex/metal/` (`mlx_accelerator.py`, `mlx_converter.py`, `memory_pool.py`, `performance_profiler.py`) and is used by `cortex/inference_engine.py` and `cortex/model_manager.py`.

## How Cortex Uses MLX

- **Generation** runs through `mlx_lm` (`generate` / `stream_generate`).
- **Acceleration** (`MLXAccelerator`) applies JIT compilation (`mx.compile`), operation fusion, AMX-friendly weight layouts, and a rotating KV cache. The KV cache size is wired to `context_length` from `config.yaml`.
- **Conversion** (`MLXConverter`) turns HuggingFace models into MLX format, cached under `~/.cortex/mlx_models`. This happens automatically on load when `mlx_backend: true` (the default); `mlx-community` models skip conversion.

## Quantization Recipes

Conversion picks a `QuantizationRecipe` based on model size and `gpu_optimization_level`:

| Recipe | Bits | Tradeoff |
|---|---|---|
| `SPEED_4BIT` | 4 | Maximum speed, largest size reduction (default for `gpu_optimization_level: maximum`) |
| `BALANCED_5BIT` | 5 | Balanced speed and quality |
| `QUALITY_8BIT` | 8 | Higher quality, moderate size reduction |
| `MIXED_PRECISION` | per-layer | Critical layers (embeddings, output head) at higher precision |
| `NONE` | 16 | No quantization |

Quantization is group-wise (`group_size: 64` by default) to limit quality loss.

## Programmatic Conversion

```python
from cortex.metal.mlx_converter import MLXConverter, ConversionConfig, QuantizationRecipe

converter = MLXConverter()
success, message, output_path = converter.convert_model(
    "some-org/some-model",
    config=ConversionConfig(quantization=QuantizationRecipe.SPEED_4BIT, group_size=64),
)
```

Converted models and metadata are cached; delete `~/.cortex/mlx_models` to force re-conversion.

## Accelerator Configuration

`MLXAccelerator` is configured internally from `config.yaml`; the useful presets are also exposed:

```python
from cortex.metal.mlx_accelerator import MLXAccelerator, MLXConfig

accelerator = MLXAccelerator(MLXConfig(**MLXAccelerator.OPTIMIZATION_PRESETS["speed"]))
```

Presets: `"speed"` (bfloat16, graph + stream parallel), `"memory"` (reduced features), `"balanced"` (float32).

## Debugging

```python
import mlx.core as mx
print(mx.default_device())   # should show Device(gpu, 0)
```

- **Slow first inference** is normal: JIT compilation happens on the first pass and results are cached.
- **Memory errors with large models**: use a smaller model or a lower-bit conversion; check `/gpu` for available memory.
- **Quantization quality issues**: use `QUALITY_8BIT` or `MIXED_PRECISION`, or adjust `group_size`.

## Measuring Performance

Use `/benchmark` inside Cortex for tokens/second and first-token latency on your machine, or `MLXAccelerator.profile_model()` / `PerformanceProfiler` programmatically. Actual throughput varies with model size, quantization, context length, and chip.

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [Apple Developer — Metal](https://developer.apple.com/metal/)
