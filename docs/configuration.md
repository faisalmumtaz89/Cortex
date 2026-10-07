# Configuration System

## Overview

Cortex reads configuration from `~/.cortex/config.yaml`. It never reads a `config.yaml` from the project it runs in, so a cloned repository's `config.yaml` cannot change Cortex's settings, such as where requests and API keys are sent. The file uses a flat key structure (no nested sections) and **every key is optional** — anything omitted falls back to the defaults in `cortex/config.py`. The `config.yaml` at the root of the Cortex repository is a commented template of the most useful keys.

Any flat key can also be overridden with a `CORTEX_<KEY>` environment variable (values parsed as YAML): `CORTEX_TOOLS_MAX_ITERATIONS=80` overrides `tools_max_iterations`, `CORTEX_TOOLS_ENABLED=false` disables tooling. Env overrides beat `config.yaml`. Unknown `CORTEX_*` variables are ignored.

Files Cortex writes outside the project:

- `~/.cortex/state.yaml` — runtime state such as the last-used model, kept separate from `config.yaml` so Cortex never rewrites your configuration.
- `~/.cortex/tool_permissions.yaml` — persisted "Allow always" tool permission rules.
- `~/.cortex/cloud_models.json` — optional additions to the cloud model catalog.

## Template

```yaml
# Inference
temperature: 0.7
top_p: 0.95
max_tokens: 32768

# Models
cloud_default_openai_model: gpt-5.5
cloud_default_anthropic_model: claude-fable-5
# cloud_azure_endpoint: https://<resource>.cognitiveservices.azure.com
# cloud_openai_compatible_base_url: https://<host>/v1

# Agent tooling
tools_enabled: true
tools_profile: full          # off | read_only | edit | full
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
- `tools_max_iterations` (default: `25`) — maximum tool-loop iterations per turn.
- `tools_idle_timeout_seconds` (default: `45`) — idle watchdog for cloud event streams.
- `tools_continue_on_reject` (default: `false`) — reserved toggle for reject handling.

### Inference

- `temperature` (default: `0.7`)
- `top_p` (default: `0.95`)
- `max_tokens` (default: `32768`) — output budget per model request, including reasoning tokens. A reply that reaches it fails the turn with a message naming the limit.

### Cloud

- `cloud_timeout_seconds` (default: `60`)
- `cloud_max_retries` (default: `2`)
- `cloud_default_openai_model` (default: `gpt-5.5`)
- `cloud_default_anthropic_model` (default: `claude-fable-5`)
- `cloud_azure_endpoint` (default: empty) — Azure OpenAI resource endpoint; `AZURE_OPENAI_ENDPOINT` env var takes precedence. Azure model ids are deployment names (`azure:<deployment>`).
- `cloud_openai_compatible_base_url` (default: empty) — base URL of an endpoint that serves the OpenAI Chat Completions API, used by `openai-compatible:<model>`; `OPENAI_COMPATIBLE_BASE_URL` env var takes precedence.

### Conversation

- `auto_save` (default: `true`) — persist conversations to `~/.cortex/conversations/conversations.db`.
- `save_directory` (default: `~/.cortex/conversations`)
- `save_format` (default: `json`), `max_conversation_history` (default: `100`), `enable_branching` (default: `true`).

### Logging

- `log_level` (default: `INFO`)
- `log_file` (default: `~/.cortex/cortex.log`)
- `log_rotation`, `max_log_size`, `performance_logging` — accepted, advisory.

### Updates

- `auto_update_check` (default: `true`) — once a day, the worker checks GitHub for a new Cortex release (a single redirect probe, in a background thread that never blocks startup) and posts a one-line notice at session start when an update exists. Set `auto_update_check: false` to opt out. `/update` always works regardless of this setting. The last check result is cached in `~/.cortex/update-check.json`.

### UI / System / Developer / Paths

Accepted for compatibility; mostly advisory in the OpenTUI runtime: `ui_theme`, `markdown_rendering`, `syntax_highlighting`, `show_*` toggles, `shutdown_timeout`, `crash_recovery`, `debug_mode`, `templates_dir`, `plugins_dir`.

## Notes

- `config.yaml` is flat; do not add nested sections like `cloud:` or `inference:`.
- Malformed tooling values are normalized rather than fatal (e.g. `tools_profile: readonly` → `read_only`, booleans coerced).
- To reset to defaults, remove `~/.cortex/config.yaml` and restart Cortex.
