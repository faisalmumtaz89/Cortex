# Configuration System

## Overview

Cortex reads configuration from `config.yaml` in the project root. The file uses a flat key structure (no nested sections). If `config.yaml` is missing, defaults from `cortex/config.py` are used.

**Runtime state:** Ephemeral values like `last_used_model`, `last_used_backend`, and `last_used_cloud_*` are stored in `~/.cortex/state.yaml`, not in `config.yaml`. This avoids polluting git diffs when you change models.

**Permission state:** Tool permission rules are stored in `~/.cortex/tool_permissions.yaml`.

**Runtime split:** The default CLI launches an OpenTUI frontend sidecar and talks to Python backend worker mode over JSON-RPC (`python -m cortex --worker-stdio`).

**Note on wiring:** Cortex accepts a broad set of configuration keys. The CLI currently **reads** these keys directly:

- `compute_backend`, `force_gpu`, `mlx_backend`, `gpu_optimization_level`
- `batch_size`, `max_batch_size`, `context_length`
- `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_tokens`, `stream_output`, `seed`
- `model_path`, `default_model`, `model_cache_dir`, `quantization_cache`
- `max_loaded_models`, `auto_quantize`, `default_quantization`, `supported_quantizations`
- `cloud_enabled`, `cloud_timeout_seconds`, `cloud_max_retries`
- `cloud_default_openai_model`, `cloud_default_anthropic_model`
- `tools_enabled`, `tools_profile`, `tools_local_mode`
- `tools_max_iterations`, `tools_idle_timeout_seconds`, `tools_continue_on_reject`
- `markdown_rendering`, `syntax_highlighting`
- `save_directory`, `auto_save`, `max_conversation_history`

Other keys are accepted but may be advisory or not yet wired in the CLI.

## Example Configuration (Excerpt)

```yaml
# GPU
compute_backend: metal
force_gpu: true
metal_performance_shaders: true
mlx_backend: true
gpu_memory_fraction: 0.9
metal_api_version: 3
shader_cache: ~/.cortex/metal_shaders
compile_shaders_on_start: true
gpu_optimization_level: maximum

# Memory
unified_memory: true
max_gpu_memory: auto
memory_pool_size: auto
kv_cache_size: 4GB
activation_memory: 2GB

# Performance
batch_size: 1
max_batch_size: 4
use_flash_attention: true
use_fused_ops: true
num_threads: 4
context_length: 8192
sliding_window_size: 4096

# Inference
temperature: 0.7
top_p: 0.95
top_k: 40
repetition_penalty: 1.1
max_tokens: 2048
stream_output: true
seed: -1

# Models
model_path: ~/models
model_cache_dir: ~/.cortex/models
quantization_cache: ~/.cortex/quantized_models
max_loaded_models: 3
verify_gpu_compatibility: true

# Cloud
cloud_enabled: true
cloud_timeout_seconds: 60
cloud_max_retries: 2
cloud_default_openai_model: gpt-5.1
cloud_default_anthropic_model: claude-sonnet-4-5

# Tooling
tools_enabled: false
tools_profile: off
tools_local_mode: disabled
tools_max_iterations: 4
tools_idle_timeout_seconds: 45
tools_continue_on_reject: false

# UI
markdown_rendering: true
show_performance_metrics: true
show_gpu_utilization: true

# Logging
log_level: INFO
log_file: ~/.cortex/cortex.log

# Conversation
auto_save: true
save_directory: ~/.cortex/conversations
max_conversation_history: 100
```

## Key Settings by Area

### GPU
- **Wired today:** `force_gpu`, `mlx_backend`, `gpu_optimization_level`  
  Other keys in this section are accepted but not currently wired in the CLI.
- `compute_backend` (default: `metal`) - Metal backend only
- `force_gpu` (default: `true`) - required for GPU-only execution
- `metal_performance_shaders` (default: `true`) - MPS acceleration for PyTorch
- `mlx_backend` (default: `true`) - MLX backend
- `gpu_memory_fraction` (default: `0.85`, config.yaml: `0.9`) - advisory fraction of available memory
- `gpu_cores` (default: `16`) - GPU cores for your chip
- `metal_api_version` (default: `3`)
- `shader_cache` (default: `~/.cortex/metal_shaders`)
- `compile_shaders_on_start` (default: `true`)
- `gpu_optimization_level` (default: `maximum`)

### Memory
- **Wired today:** memory pool auto‑sizing is internal; the keys below are currently advisory.
- `unified_memory` (default: `true`)
- `max_gpu_memory` (default: `20GB`, config.yaml: `auto`)
- `memory_pool_size` (default: `20GB`, config.yaml: `auto`)
- `kv_cache_size` (default: `2GB`, config.yaml: `4GB`)
- `activation_memory` (default: `2GB`)

### Performance
- **Wired today:** `batch_size`, `max_batch_size`, `context_length`  
  Other keys in this section are accepted but not currently wired in the CLI.
- `batch_size` (default: `8`, config.yaml: `1`)
- `max_batch_size` (default: `16`, config.yaml: `4`)
- `use_flash_attention` (default: `true`)
- `use_fused_ops` (default: `true`)
- `num_threads` (default: `1`, config.yaml: `4`)
- `context_length` (default: `32768`, config.yaml: `8192`)
- `sliding_window_size` (default: `4096`)

### Inference
- `temperature` (default: `0.7`)
- `top_p` (default: `0.95`)
- `top_k` (default: `40`)
- `repetition_penalty` (default: `1.1`)
- `max_tokens` (default: `2048`)
- `stream_output` (default: `true`)
- `seed` (default: `-1`)

### Models
- `model_path` (default: `~/models`)
- `default_model` (default: empty)
- `last_used_model` is persisted to `~/.cortex/state.yaml` (default: empty)
- `model_cache_dir` (default: `~/.cortex/models`)
- `preload_models` (default: `[]`)
- `max_loaded_models` (default: `3`)
- `lazy_load` (default: `false`)
- `verify_gpu_compatibility` (default: `true`)
- `default_quantization` (default: `Q4_K_M`)
  - Preferred quantization for MLX conversion. Accepted values include:
    - `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0` (mapped to MLX 4/5/8-bit recipes)
    - `4bit`, `5bit`, `8bit`, `mixed`, `none`, `auto`
- `supported_quantizations` (default: `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`)
  - Validation list for `default_quantization` (Q* values)
- `auto_quantize` (default: `true`)
  - Enables dynamic quantization fallback for PyTorch/SafeTensors models
- `quantization_cache` (default: `~/.cortex/quantized_models`)
  - Cache directory for dynamic-quantized PyTorch/SafeTensors models

### Cloud
- `cloud_enabled` (default: `true`)
- `cloud_timeout_seconds` (default: `60`)
- `cloud_max_retries` (default: `2`)
- `cloud_default_openai_model` (default: `gpt-5.1`)
- `cloud_default_anthropic_model` (default: `claude-sonnet-4-5`)

### Tooling
- `tools_enabled` (default: `false`)
  - Master toggle for tool execution.
- `tools_profile` (default: `off`)
  - One of `off`, `read_only`, `patch`, `full`.
- `tools_local_mode` (default: `disabled`)
  - `disabled` or `experimental` for local-model tool loops.
- `tools_max_iterations` (default: `4`)
  - Maximum tool loop iterations per turn.
- `tools_idle_timeout_seconds` (default: `45`)
  - Idle timeout used by cloud event streaming watchdog.
- `tools_continue_on_reject` (default: `false`)
  - Reserved behavior toggle for reject handling (conservative default).

### UI
- **Wired today:** `markdown_rendering`, `syntax_highlighting` (CLI)  
  Other UI keys are used only by the Textual UI (not the default CLI).
- `ui_theme` (default: `default`)
- `syntax_highlighting` (default: `true`)
- `markdown_rendering` (default: `true`)
- `show_performance_metrics` (default: `true`)
- `show_gpu_utilization` (default: `true`)
- `auto_scroll` (default: `true`)
- `copy_on_select` (default: `true`)
- `mouse_support` (default: `true`)

### Logging
- **Wired today:** logging is handled internally; keys below are currently advisory.
- `log_level` (default: `INFO`)
- `log_file` (default: `~/.cortex/cortex.log`)
- `log_rotation` (default: `daily`)
- `max_log_size` (default: `100MB`)
- `performance_logging` (default: `true`)
- `gpu_metrics_interval` (default: `1000` ms)

### Conversation
- `auto_save` (default: `true`)
- `save_format` (default: `json`)
- `save_directory` (default: `~/.cortex/conversations`)
- `max_conversation_history` (default: `100`)
- `enable_branching` (default: `true`)

### System
- **Wired today:** `auto_update_check` is honored at startup for installer/package installs; other keys below remain advisory.
- `startup_checks` (default: `verify_metal_support`, `check_gpu_memory`, `validate_models`, `compile_shaders`)
- `shutdown_timeout` (default: `5`)
- `crash_recovery` (default: `true`)
- `auto_update_check` (default: `false`, but update checks run unless explicitly set to `false`)

### Developer
- **Wired today:** developer toggles are currently advisory.
- `debug_mode` (default: `false`)
- `profile_inference` (default: `false`)
- `metal_capture` (default: `false`)
- `verbose_gpu_logs` (default: `false`)

### Paths
- **Wired today:** template and plugin paths are used by the template registry; other keys are advisory.
- `templates_dir` (default: `~/.cortex/templates`)
- `plugins_dir` (default: `~/.cortex/plugins`)

## Notes

- `config.yaml` is flat; do not add nested sections like `gpu:` or `inference:`.
- If you want to reset to defaults, remove `config.yaml` and restart Cortex.
