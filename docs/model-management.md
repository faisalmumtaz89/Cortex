# Model Management

## Overview

Cortex handles local model discovery, loading, MLX conversion, and caching. The logic lives in `cortex/model_manager.py`. Local inference runs on two backends only:

- **MLX** (recommended): native Apple Silicon models — a directory with `config.json` plus `*.safetensors` or `weights.npz`, loaded via `mlx_lm`.
- **GGUF**: llama.cpp models — a single `.gguf` file, loaded via `llama-cpp-python` with Metal.

Anything else (bare PyTorch checkpoints, GPTQ/AWQ-quantized models) is not loadable directly; loading such a model fails with a message pointing at MLX conversion or a GGUF variant.

## Format Detection

- `*.gguf` file → **GGUF**
- Directory with `config.json` + (`weights.npz` or `*.safetensors`) → **MLX** (plain HuggingFace safetensors directories qualify and load through `mlx_lm`)
- Directory with `pytorch_model*.bin`, or safetensors flagged GPTQ/AWQ in `config.json` → detected but **unsupported**

## MLX Conversion and Cache

When `mlx_backend` is enabled (the default), Cortex auto-converts non-MLX HuggingFace models to MLX for faster inference:

- Converted outputs are cached in `~/.cortex/mlx_models`
- Conversion uses `MLXConverter` (`cortex/metal/mlx_converter.py`) with a quantization recipe chosen by model size
- `gpu_optimization_level: maximum` prefers a 4-bit speed-optimized conversion
- `mlx-community/<name>` repo ids passed to `/model` are downloaded and loaded directly

If a model is already in MLX format, conversion is skipped.

## Quantization Metadata

Model metadata tracks quantization via `QuantizationType`: `NONE`, `INT4`, `INT8`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `FP16`, `FP32` (plus detected-but-unsupported `GPTQ`/`AWQ`). MLX conversion recipes are 4/5/8-bit or mixed precision; GGUF quantization is whatever the file was built with.

## Loading Behavior

- `/model <selector>` loads by list number, name, path, or `provider:model` (cloud).
- Up to `max_loaded_models` (default 3) local models stay resident; the oldest is unloaded past the limit.
- GPU compatibility (available unified memory vs model size) is checked before loading.
- The last used model is persisted to `~/.cortex/state.yaml` and restored on the next start.

## Downloads

`/download <repo_id> [filename] [--load]` pulls models from HuggingFace into `model_path` (default `~/models`):

- Repository snapshot downloads (MLX models) or single-file downloads (GGUF).
- Interrupted snapshots are detected and resumed on retry.
- Progress is streamed into the TUI; downloads can be cancelled.
- Gated repos require HuggingFace auth (`huggingface-cli login` in your shell; check with `/login huggingface`).

## Model Info and Listing

`/model` opens an interactive picker showing local models (with active/loaded tags) and the cloud catalog; `/model list` prints the same as text. Internally, `ModelInfo` captures format, quantization, size, parameter count, context length, and GPU memory usage.
