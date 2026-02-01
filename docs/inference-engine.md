# Inference Engine

## Overview

The Cortex inference engine runs GPU-only text generation on Apple Silicon. It routes requests to the correct backend based on the loaded model format and tracks performance metrics.

Backends:
- **MLX** for native Apple Silicon models
- **PyTorch MPS** for Hugging Face models
- **GGUF** via llama-cpp-python

## Core Classes

### InferenceEngine

Responsibilities:
- Validate GPU backend availability (MPS + MLX)
- Initialize Metal optimizations (memory pool, MPS optimizer, MLX accelerator)
- Route generation by model format
- Track generation status and metrics

Statuses:
- `IDLE`, `LOADING`, `GENERATING`, `COMPLETED`, `ERROR`, `CANCELLED`

### GenerationRequest

Fields:
- `prompt` (str)
- `max_tokens` (int)
- `temperature` (float)
- `top_p` (float)
- `top_k` (int)
- `repetition_penalty` (float)
- `stop_sequences` (list[str])
- `stream` (bool)
- `seed` (int | None)

### GenerationMetrics

Tracks:
- tokens generated
- total time
- tokens/sec
- first token latency
- GPU utilization (CPU‑based proxy; not hardware GPU telemetry)
- memory usage

## Supported Model Formats

Generation is routed by `ModelFormat`:
- **MLX**: `mlx_lm` generation (preferred)
- **SafeTensors**: PyTorch MPS path
- **PyTorch**: PyTorch MPS path
- **Quantized**: PyTorch path (GPTQ/AWQ models)
- **GGUF**: llama-cpp-python path

## Streaming Behavior

- **MLX**: streams tokens when `mlx_lm.stream_generate` is available
- **PyTorch**: uses `TextIteratorStreamer` for streaming
- **GGUF**: streams through llama-cpp-python

## Template Handling

Prompt formatting and response filtering live in the Template Registry (see `docs/template-registry.md`) and are applied in the CLI layer. The inference engine focuses on generation only.

## GPU Validation

On startup the engine verifies:
- MPS is available (`torch.backends.mps.is_available()`)
- MLX can access the GPU (`mx.default_device()`)

If either fails, Cortex exits with a clear error.

Note: The engine requires both MPS and MLX to be available; this is enforced on startup.
