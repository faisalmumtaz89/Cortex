# Command Line Interface

## Overview

Cortex is a terminal AI coding agent. `cortex` launches the interactive TUI; `cortex -p "..."` runs a single agent turn headlessly. All model and session management happens through slash commands.

## Interactive Mode

```bash
cortex
```

This launches the OpenTUI frontend (single terminal writer), which spawns the Python backend in worker mode (`python -P -m cortex --worker-stdio`; `-P` keeps the project directory off its import path) and talks to it over line-delimited JSON-RPC 2.0.

If the OpenTUI sidecar is unavailable in a source checkout, run `./install.sh` at the repository root to build and install it.

Type a message to run an agent turn with the active model, or a slash command to manage the session.

### Keyboard

| Key | Action |
|---|---|
| `Enter` | Submit input |
| `Shift+Enter` | Insert newline |
| `1` / `2` / `3` / `Esc` | Answer a pending permission prompt |
| `Ctrl+C` | Exit Cortex |

## Headless Mode

```bash
cortex -p "PROMPT" [--model <selector>] [--full-auto]
```

Runs one agent turn through the same worker wiring the TUI uses:

- The assistant reply streams to **stdout**; tool activity and errors go to **stderr**, so stdout stays pipeable.
- `--model` accepts the same `provider:model` selectors as `/model`.
- Permission policy: reads (`read_file`, `list_dir`, `search`) are allowed; `edit_file`, `write_file`, and `bash` are denied unless `--full-auto` is passed (there is no interactive prompt). Persisted rules from `~/.cortex/tool_permissions.yaml` still apply.
- Exit codes: `0` success, `1` turn error, `2` setup error (e.g. model selection failed).

Examples:

```bash
cortex -p "summarize what cortex/app/headless.py does"
cortex -p "rename the helper in src/utils.py and update callers" --full-auto
cortex -p "review this diff for bugs" --model openai:gpt-5.1
```

## Slash Commands

| Command | Description |
|---|---|
| `/help` | List available commands |
| `/status` | Active model and the model that answered the last turn |
| `/model [provider:model]` | Pick a model interactively, or switch by `provider:model` |
| `/login <provider> [api_key]` | Manage OpenAI, Anthropic, Azure, and OpenAI-compatible credentials |
| `/update [cortex]` | Show installed vs latest version, or update Cortex |
| `/clear` | Clear conversation history |
| `/save` | Save the conversation as JSON |
| `/quit` or `/exit` | Exit Cortex |

### `/model` — switch models

```bash
/model                                  # open the interactive picker (↑↓ + Enter, Esc cancels)
/model openai:gpt-5.1
/model anthropic:claude-sonnet-4-5
/model openai-compatible:deepseek/deepseek-v4.1-flash
/model list                             # plain text list (headless/worker fallback)
```

### `/login` — credentials

```bash
/login openai <api_key>       # store an OpenAI key
/login anthropic <api_key>    # store an Anthropic key
/login azure <api_key>        # store an Azure OpenAI key
/login openai-compatible <api_key>  # store a key for an OpenAI-compatible endpoint
/login openai                 # show auth status for a provider
```

`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `AZURE_OPENAI_API_KEY` environment
variables are used as fallbacks. Azure additionally requires the resource
endpoint via `AZURE_OPENAI_ENDPOINT` (or `cloud_azure_endpoint` in
`~/.cortex/config.yaml`); Azure model ids are your deployment names, selected as
`/model azure:<deployment>` (e.g. `azure:gpt-5.5`).

Any other endpoint that serves the OpenAI Chat Completions API (DeepSeek,
OpenRouter, vLLM, and similar) is reached through the `openai-compatible`
provider. Set its base URL via `OPENAI_COMPATIBLE_BASE_URL` (or
`cloud_openai_compatible_base_url` in `~/.cortex/config.yaml`), provide the key with
`/login openai-compatible <api_key>` or `OPENAI_COMPATIBLE_API_KEY`, and select
the model as `/model openai-compatible:<model>`.


## Agent Tools and Permissions

With tools enabled (the default: `tools_enabled: true`, `tools_profile: full`), the model can call:

- `read_file`, `list_dir`, `search` — read-only, auto-allowed
- `edit_file` (exact string replacement), `write_file` — prompt for permission
- `bash` — prompts for permission

Every tool is sandboxed to the directory Cortex was started in. When permission is needed, the TUI shows an arrow menu (↑↓ to choose, Enter to confirm):

- **Allow once** — remembered for the current session
- **Allow always** — persisted to `~/.cortex/tool_permissions.yaml`
- **Reject** (or `Esc`) — the model continues without the tool result

Profiles restrict the exposed tool set: `off` (none), `read_only`, `edit` (adds `edit_file`/`write_file`), `full` (adds `bash`). See `docs/configuration.md` for the `tools_*` keys.

## Configuration

Cortex reads an optional `~/.cortex/config.yaml`. Common keys: `temperature`, `max_tokens`, `tools_profile`, `cloud_default_openai_model`. See the [Configuration Guide](configuration.md).
