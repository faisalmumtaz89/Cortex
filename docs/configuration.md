# Configuration System

## Lumen (local inference)

- `lumen_binary` / `lumen_server_binary` (default: `lumen` / `lumen-server` on PATH)
- `lumen_port` (default `0` = pick a free port)
- `lumen_context_len` (default `0` = Lumen's default, 8192)
- `lumen_startup_timeout_seconds` (default `180`)
- `lumen_log_level` (default `warn`)

## Overview

Cortex reads configuration from `config.yaml` in the directory it starts in. The file uses a flat key structure (no nested sections) and **every key is optional** — anything omitted falls back to the defaults in `cortex/config.py`. The repository's `config.yaml` is a commented template of the most useful keys.

Any flat key can also be overridden with a `CORTEX_<KEY>` environment variable (values parsed as YAML): `CORTEX_TOOLS_MAX_ITERATIONS=80` overrides `tools_max_iterations`, `CORTEX_TOOLS_ENABLED=false` disables tooling. Env overrides beat `config.yaml`. Unknown `CORTEX_*` variables are ignored.

Files Cortex writes outside the project:

- `~/.cortex/state.yaml` — runtime state such as `last_used_model` and the last-used backend, kept out of `config.yaml` so switching models does not pollute git diffs.
- `~/.cortex/tool_permissions.yaml` — persisted "Allow always" tool permission rules.
- `~/.cortex/cloud_models.json` — optional additions to the cloud model catalog.

## Template

```yaml
# Inference
temperature: 0.7
top_p: 0.95
max_tokens: 4096
context_length: 8192

# Local models

# Cloud models
cloud_default_openai_model: gpt-5.1
cloud_default_anthropic_model: claude-sonnet-4-5
# cloud_azure_endpoint: https://<resource>.cognitiveservices.azure.com

# Agent tooling
tools_enabled: true
tools_profile: full          # off | read_only | edit | full
tools_local_mode: experimental
tools_max_iterations: 25

# Logging
log_level: INFO
log_file: ~/.cortex/cortex.log
```

## Key Reference

### Agent tooling

- `tools_enabled` (default: `true`) — master toggle for tool execution.
- `tools_profile` (default: `full`) — which tools the model may call:
  - `off`: none
  - `read_only`: `read_file`, `list_dir`, `search`
  - `edit`: read-only + `edit_file`, `write_file` (the legacy value `patch` is accepted as an alias)
  - `full`: edit + `bash`
- `tools_local_mode` (default: `experimental`) — `experimental` enables the `<tool_calls>` JSON protocol for local models; `disabled` restricts tools to cloud models.
- `tools_max_iterations` (default: `25`) — maximum tool-loop iterations per turn.
- `tools_idle_timeout_seconds` (default: `45`) — idle watchdog for cloud event streams.
- `tools_continue_on_reject` (default: `false`) — reserved toggle for reject handling.

### Inference

- `temperature` (default: `0.7`)
- `top_p` (default: `0.95`)
- `top_k` (default: `40`)
- `repetition_penalty` (default: `1.1`)
- `max_tokens` (default: `4096`)
- `stream_output` (default: `true`)
- `seed` (default: `-1`)

### Models

- `default_model` (default: empty) — model to load on startup; otherwise the last-used model is restored from `~/.cortex/state.yaml`.
- `max_loaded_models` (default: `3`) — oldest model is unloaded past this limit.
- `verify_gpu_compatibility` (default: `true`)

### Cloud

- `cloud_enabled` (default: `true`)
- `cloud_timeout_seconds` (default: `60`)
- `cloud_max_retries` (default: `2`)
- `cloud_default_openai_model` (default: `gpt-5.1`)
- `cloud_default_anthropic_model` (default: `claude-sonnet-4-5`)
- `cloud_azure_endpoint` (default: empty) — Azure OpenAI resource endpoint; `AZURE_OPENAI_ENDPOINT` env var takes precedence. Azure model ids are deployment names (`azure:<deployment>`).

### Performance

- `context_length` (default: `32768`; the template sets `8192`)
- `batch_size` (default: `8`), `max_batch_size` (default: `16`)
- `use_flash_attention`, `use_fused_ops`, `num_threads`, `sliding_window_size` — accepted, largely advisory for the MLX/GGUF backends.

### GPU

The Metal backend is mandatory; these keys mostly tune conversion and are otherwise advisory:

- `mlx_backend` (default: `true`) — auto-convert non-MLX HuggingFace models to MLX on load.
- `gpu_optimization_level` (default: `maximum`) — `maximum` prefers speed-optimized 4-bit MLX conversion.
- `compute_backend` (`metal` only), `force_gpu` (`true` only), `gpu_memory_fraction`, `gpu_cores`, `metal_api_version`, `shader_cache`, `compile_shaders_on_start`.

### Memory

Advisory hints: `unified_memory`, `max_gpu_memory`, `memory_pool_size`, `kv_cache_size`, `activation_memory`. `cpu_offload` must remain `false`.

### Conversation

- `auto_save` (default: `true`) — persist conversations to `~/.cortex/conversations/conversations.db`.
- `save_directory` (default: `~/.cortex/conversations`)
- `save_format` (default: `json`), `max_conversation_history` (default: `100`), `enable_branching` (default: `true`).

### Logging

- `log_level` (default: `INFO`)
- `log_file` (default: `~/.cortex/cortex.log`)
- `log_rotation`, `max_log_size`, `performance_logging`, `gpu_metrics_interval` — accepted, advisory.

### UI / System / Developer / Paths

Accepted for compatibility; mostly advisory in the OpenTUI runtime: `ui_theme`, `markdown_rendering`, `syntax_highlighting`, `show_*` toggles, `startup_checks`, `shutdown_timeout`, `crash_recovery`, `auto_update_check`, `debug_mode`, `profile_inference`, `metal_capture`, `verbose_gpu_logs`, `templates_dir`, `plugins_dir`.

## Notes

- `config.yaml` is flat; do not add nested sections like `gpu:` or `inference:`.
- Malformed tooling values are normalized rather than fatal (e.g. `tools_profile: readonly` → `read_only`, booleans coerced).
- To reset to defaults, remove `config.yaml` and restart Cortex.
