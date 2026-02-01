# Dynamic Quantization

## Overview

Dynamic quantization is a fallback path for PyTorch and SafeTensors models that do not fit in available GPU memory. When triggered, Cortex quantizes linear layers to INT8 or INT4 and keeps the remaining weights in FP16 to reduce memory usage while preserving quality.

## When It Runs

Dynamic quantization is only applied when all of the following are true:

- The model format is **PyTorch** or **SafeTensors**
- The model is **not already quantized** (GPTQ, AWQ, INT4/INT8)
- The GPU compatibility check fails for the current available memory
- `auto_quantize` is enabled in `config.yaml`

It is **not** used for MLX or GGUF models.

## How It Chooses INT8 vs INT4

- **INT8 is preferred** if it will fit in memory.
- **INT4 is used** only when INT8 still exceeds available memory.
- Very small models avoid INT4 when possible to preserve quality.
- If INT4 validation fails, Cortex falls back to INT8.

## What Gets Quantized

- **Linear layers** are quantized (per-channel, symmetric by default).
- All other parameters remain in **FP16**.

## Cache Behavior

Quantized weights are cached to disk for faster reloads.

- Default cache: `~/.cortex/quantized_models`
- Config key: `quantization_cache`

Delete the cache directory to force re-quantization.

## How to Control It

There is no explicit CLI toggle for dynamic quantization today. If you need more control:

- Disable it with `auto_quantize: false` in `config.yaml`.
- Convert the model to **MLX** and select a quantization recipe.
- Use a smaller model size.
- Free additional system memory before loading.

## Tradeoffs

- **INT8** preserves quality better and is the default choice when possible.
- **INT4** is a last-resort fallback for very large models.
