# Command Line Interface

## Overview

Cortex is a terminal AI coding agent. `cortex` launches the interactive TUI; `cortex -p "..."` runs a single agent turn headlessly. All model and session management happens through slash commands.

## Interactive Mode

```bash
cortex
```

This launches the OpenTUI frontend (single terminal writer), which spawns the Python backend in worker mode (`python -m cortex --worker-stdio`) and talks to it over line-delimited JSON-RPC 2.0.

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
- `--model` accepts the same selectors as `/model` (local name/path or `provider:model`).
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
| `/status` | Current setup (GPU, model, settings) |
| `/gpu` | GPU and memory details |
| `/model [selector]` | Pick a model interactively, or switch by name / `provider:model` |
| `/download <model[:quant]>` | Download a local model via Lumen |
| `/setup` | Load the first available local model if none is active |
| `/benchmark [tokens] [--prompt <text>]` | Performance test (local models only) |
| `/login <provider> [api_key]` | Manage OpenAI/Anthropic/Azure credentials |
| `/update [lumen\|cortex]` | Show installed vs latest versions, or update the Lumen engine / Cortex itself |
| `/clear` | Clear conversation history |
| `/save` | Save the conversation as JSON |
| `/quit` or `/exit` | Exit Cortex |

### `/model` — switch models

```bash
/model                                  # open the interactive picker (↑↓ + Enter, Esc cancels)
/model nanbeige                         # local model by (unambiguous) name or prefix
/model ~/models/My-Model-4bit           # local model by path
/model openai:gpt-5.1                   # cloud model
/model anthropic:claude-sonnet-4-5
/model list                             # plain text list (headless/worker fallback)
```

### `/download` — fetch local models via Lumen

```bash
/download qwen3-5-9b:q4_0
/download qwen3-6-27b            # default quant Q8_0
/download cancel                 # cancel an in-flight download
```

- Local models are downloaded and converted by the Lumen engine (`lumen pull`); only Lumen-supported models are available — the `/model` picker marks them `download required`.
- Progress streams into the TUI; after it finishes, load with `/model <model:quant>`.


### `/login` — credentials

```bash
/login openai <api_key>       # validate and store an OpenAI key
/login anthropic <api_key>    # validate and store an Anthropic key
/login azure <api_key>        # store an Azure OpenAI key
/login openai                 # show auth status for a provider
```

`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `AZURE_OPENAI_API_KEY` environment
variables are used as fallbacks. Azure additionally requires the resource
endpoint via `AZURE_OPENAI_ENDPOINT` (or `cloud_azure_endpoint` in
`config.yaml`); Azure model ids are your deployment names, selected as
`/model azure:<deployment>` (e.g. `azure:gpt-5.5`).


### `/benchmark` — local performance test

```bash
/benchmark
/benchmark 200 --prompt "Once upon a time"
```

Reports tokens/second, first-token latency, and memory usage. Cloud models are not benchmarked.

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

Cortex reads an optional `config.yaml` from the directory it starts in. Common keys: `model_path`, `default_model`, `temperature`, `max_tokens`, `tools_profile`. See the [Configuration Guide](configuration.md).
