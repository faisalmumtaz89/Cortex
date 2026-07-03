# Inference Engine

## Overview

The inference engine (`cortex/inference_engine.py`) runs GPU-only text generation on Apple Silicon. It routes requests to the correct backend based on the loaded model format and tracks performance metrics.

Backends:

- **MLX** via `mlx_lm` (primary)
- **GGUF** via `llama-cpp-python` (Metal)

## Core Classes

### InferenceEngine

Responsibilities:

- Initialize Metal optimizations (memory pool, MLX accelerator, performance profiler)
- Route generation by model format (MLX or GGUF)
- Track generation status and metrics
- Provide `/benchmark`

Statuses: `IDLE`, `LOADING`, `GENERATING`, `COMPLETED`, `ERROR`, `CANCELLED`.

### GenerationRequest

Fields: `prompt`, `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop_sequences`, `stream`, `seed`.

### GenerationMetrics

Tracks tokens generated, total time, tokens/sec, first-token latency, GPU utilization (a CPU-based proxy, not hardware GPU telemetry), and memory usage.

## Streaming

- **MLX**: streams tokens via `mlx_lm.stream_generate`
- **GGUF**: streams through llama-cpp-python

## Template Handling

Prompt formatting and response filtering for local models live in the Template Registry (see `docs/template-registry.md`). For agent turns, the system prompt and tool protocol are assembled by `cortex/tooling/agent_prompt.py`. The inference engine focuses on generation only.

## GPU Requirements

Startup validation (`cortex/gpu_validator.py`) requires Apple Silicon with Metal support and the MLX framework. In worker and headless modes validation is non-strict: a failure degrades local inference but still allows cloud-only use.
