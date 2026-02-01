# Model Management

## Overview

Cortex handles model discovery, loading, conversion, quantization, and caching for local LLM inference on Apple Silicon. The logic lives in `cortex/model_manager.py`.

## Supported Model Formats

- **MLX** (recommended): native Apple Silicon format (`weights.npz` or `model.safetensors` + `config.json`)
- **GGUF**: llama.cpp models (`.gguf` file)
- **SafeTensors**: HuggingFace models with `*.safetensors`
- **PyTorch**: HuggingFace models without SafeTensors
- **Quantized (GPTQ/AWQ)**: detected via `config.json` or weight tensor naming (optional deps required)

Note: Some GPTQ/AWQ loaders do not support MPS directly and may fall back to CPU execution.

## Auto‑Detection Rules

Cortex detects model format using the filesystem structure:

- `*.gguf` file → **GGUF**
- Directory with `weights.npz` → **MLX**
- Directory with `*.safetensors` → **SafeTensors**
- Directory with `config.json` only → **PyTorch**

Quantized formats (GPTQ/AWQ/INT4/INT8) are detected from `config.json` or tensor names when present.

## MLX Conversion and Cache

When MLX is enabled (default), Cortex auto‑converts non‑MLX models to MLX for faster inference:

- Cached outputs live in `~/.cortex/mlx_models`
- Conversion uses `MLXConverter` and a quantization recipe based on model size
- `gpu_optimization_level: maximum` forces a 4‑bit speed‑optimized conversion

If a model is already in MLX format, conversion is skipped and it is loaded directly.

## Dynamic Quantization Fallback (PyTorch/SafeTensors)

When a PyTorch or SafeTensors model is too large for available GPU memory, Cortex can apply dynamic quantization:

- **INT8 is preferred**
- **INT4 is used only when INT8 still does not fit**
- Cached quantized weights are stored in `~/.cortex/quantized_models`

See `docs/dynamic-quantization.md` for details.

## Quantization Types

Model metadata tracks quantization using `QuantizationType`:

- `NONE`, `INT4`, `INT8`, `GPTQ`, `AWQ`
- `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0` (GGUF / llama.cpp style)
- `FP16`, `FP32`

MLX conversion uses 4/5/8‑bit recipes and reports INT4/INT8 in model metadata.

## Fine‑Tuned Models

LoRA‑fine‑tuned models are stored as MLX folders under:

```
~/.cortex/mlx_models/<model_name>
```

Detection uses:

- `adapter.safetensors`
- `fine_tuned.marker` (when present)

Fine‑tuned models appear in `/model` just like normal models.

## Model Info and Listing

The CLI shows model size, format, and quantization where available. Internally, `ModelInfo` captures:

- Format and quantization
- Model size and parameters
- Context length
- GPU memory usage

Use `/model` to load or manage models and `/download` to pull models from HuggingFace.
