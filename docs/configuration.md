# Configuration System

## Overview

Cortex reads configuration from `~/.cortex/config.yaml`. It never reads a `config.yaml` from the project it runs in, so a cloned repository's `config.yaml` cannot change Cortex's settings, such as where requests and API keys are sent. The file uses a flat key structure (no nested sections) and **every key is optional** — anything omitted falls back to its default. The [Template](#template) below lists the most useful keys.

A setting that is unknown or not valid (for example a number above its maximum) is ignored, and the others still apply: it falls back to its default, or, for a `CORTEX_*` variable that is not valid, to the value in `config.yaml`. Cortex lists what it ignored in the notice when a session starts (on stderr for a headless `-p` run) and in `~/.cortex/cortex.log`. `tools_idle_timeout_seconds` is not read: `cloud_timeout_seconds` sets the request timeout, and a permission prompt waits for your answer.

Any flat key can also be overridden with a `CORTEX_<KEY>` environment variable (values parsed as YAML): `CORTEX_TOOLS_MAX_ITERATIONS=80` overrides `tools_max_iterations`, `CORTEX_TOOLS_ENABLED=false` disables tooling. Env overrides beat `config.yaml`. Unknown `CORTEX_*` variables are ignored.

Files Cortex writes outside the project:

- `~/.cortex/state.yaml` — runtime state, kept separate from `config.yaml` so Cortex never rewrites your configuration: the last-used model, and the Azure endpoint and OpenAI-compatible base URL last used to select a model, so later sessions work without setting them again. `AZURE_OPENAI_ENDPOINT`/`OPENAI_COMPATIBLE_BASE_URL` and the `config.yaml` keys take precedence over the saved ones.
- `~/.cortex/credentials.json` — API keys saved with `/login` where no system keyring works, readable only by you.
- `~/.cortex/cortex.log` — the log; `~/.cortex/update-check.json` — when Cortex last checked for a new release.
- `~/.cortex/conversations/` — conversation history (`conversations.db`) and conversations saved with `/save`.
- A temporary folder `cortex-tui-<pid>-…` in your temp directory, where the TUI unpacks its native library; it is removed when Cortex exits (or at the next start, after a crash).
- `~/.cortex/tool_permissions.yaml` — persisted tool permission rules. "Allow always" adds a rule for one exact command or one file, in the directory Cortex was started in (`repo:`). A command run in another directory is saved with it, as `cd ./<dir> || exit 1` (the directory quoted when its name needs it) and the command on the next line. File patterns are paths relative to the start directory, such as `src/app.py`. Saved patterns escape wildcard characters, so `ls *.py` is saved as `ls [*].py` and matches only itself. A call with no command or file to name is never saved. Changes to the file apply at once, also in running sessions; while the file cannot be parsed, sessions keep the rules they last read. A rule you write without `repo:` applies everywhere; `repo:` takes an absolute path (or one starting with `~`). When several rules match, the last one in the file wins. "Allow always" answers this session could not save still apply until it ends. Its `pattern` may use the wildcards `*`, `?` and `[...]`, but a wildcard in a `bash` pattern matches anything, including further commands: `git *` also allows `git log; rm -rf ~`. Files with `version: 1` were written by older Cortex versions, whose "Allow always" covered every command or file in every repository: their allow rules for `bash` and edits are ignored and removed at the next save, so those tools ask again.
- `~/.cortex/cloud_models.json` — optional models to add to the `/model` picker, by provider, such as `{"openai-compatible": ["deepseek/deepseek-v4.1-flash"], "azure": ["my-deployment"]}`. `openai-compatible` has no built-in models, so this file is how its models appear in the picker.

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
tools_max_iterations: 100

# Logging
log_level: INFO
log_file: ~/.cortex/cortex.log
```

## Key Reference

### Agent tooling

- `tools_enabled` (default: `true`) — master toggle for tool execution.
- `tools_profile` (default: `full`) — which tools the model may call:
  - `off`: none
  - `read_only`: `read_file`, `list_dir`, `search` (and `skill` when skills exist)
  - `edit`: read-only + `edit_file`, `write_file` (the legacy value `patch` is accepted as an alias)
  - `full`: edit + `bash`
- `tools_max_iterations` (default: `100`, at most `1000`) — model requests one turn may make while it uses tools. A turn that reaches the limit stops with a message naming it; its tool calls and results stay in the conversation, so a new message lets the model continue.
- `tools_continue_on_reject` (default: `false`) — not read.

### Inference

- `temperature` (default: `0.7`, 0 to 2)
- `top_p` (default: `0.95`, 0 to 1)
- `max_tokens` (default: `32768`) — output budget per model request, including reasoning tokens. A reply that reaches it fails the turn with a message naming the limit.

### Cloud

- `cloud_timeout_seconds` (default: `60`, at most `600`) — seconds a model request may wait for data before it fails (and is sent again, see `cloud_max_retries`). A permission prompt has no timeout: it waits for your answer, and interrupting the turn rejects it.
- `cloud_max_retries` (default: `2`, at most `10`) — how many times a model request is sent again after a connection error, a timeout, a stream cut off before its end, a rate limit (429), a request timeout or conflict (408, 409), or a server error (5xx, or an overloaded or server error reported inside the stream). Cortex waits as the server asks (`Retry-After` or `retry-after-ms`, at most 60 seconds), otherwise 1, 2, 4, ... seconds (at most 30). Other errors, such as an invalid request (400) or a rejected key (401), fail the turn at once. A request whose reply has started to appear is not sent again.
- `cloud_default_openai_model` (default: `gpt-5.5`), `cloud_default_anthropic_model` (default: `claude-fable-5`) — the model Cortex selects at startup when no saved model can be restored and that provider has a key.
- `cloud_azure_endpoint` (default: empty) — Azure OpenAI resource endpoint; `AZURE_OPENAI_ENDPOINT` env var takes precedence. Azure model ids are deployment names (`azure:<deployment>`).
- `cloud_openai_compatible_base_url` (default: empty) — base URL of an endpoint that serves the OpenAI Chat Completions API, used by `openai-compatible:<model>`; `OPENAI_COMPATIBLE_BASE_URL` env var takes precedence.

### Conversation

- `auto_save` (default: `true`) — persist conversations to `~/.cortex/conversations/conversations.db`.
- `save_directory` (default: `~/.cortex/conversations`)
- `max_conversation_history` (default: `100`).
- `save_format` (default: `json`) and `enable_branching` (default: `true`) — not read.

### Logging

- `log_level` (default: `INFO`)
- `log_file` (default: `~/.cortex/cortex.log`)
- `log_rotation`, `max_log_size`, `performance_logging` — accepted, advisory.

### Updates

- `auto_update_check` (default: `true`) — once a day, the worker checks GitHub for a new Cortex release (a single redirect probe, in a background thread that never blocks startup) and posts a one-line notice at session start when an update exists. Set `auto_update_check: false` to opt out. `/update` always works regardless of this setting. The last check result is cached in `~/.cortex/update-check.json`.

### UI / System / Developer / Paths

Accepted but not read: `ui_theme`, `markdown_rendering`, `syntax_highlighting`, `show_performance_metrics`, `auto_scroll`, `copy_on_select`, `mouse_support`, `shutdown_timeout`, `crash_recovery`, `templates_dir`, `plugins_dir`.

`debug_mode: true` also writes the log to stderr.

## Notes

- `config.yaml` is flat; do not add nested sections like `cloud:` or `inference:`.
- Malformed tooling values are normalized rather than fatal (e.g. `tools_profile: readonly` → `read_only`, booleans coerced).
- To reset to defaults, remove `~/.cortex/config.yaml` and restart Cortex.
