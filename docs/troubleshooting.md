# Troubleshooting Guide

## How Cortex Works

Cortex uses a split runtime:

- `cortex` launches an OpenTUI frontend sidecar (terminal renderer)
- the frontend talks to the Python backend worker (`python -P -m cortex --worker-stdio`) over JSON-RPC
- `cortex -p "..."` runs one headless agent turn through the same worker wiring

**Available slash commands:**

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/status` | Show the active model and the model that answered the last turn |
| `/model` | List and switch models |
| `/login` | Manage OpenAI, Anthropic, Azure, and OpenAI-compatible credentials |
| `/update` | Show the installed version and any newer one, or update Cortex |
| `/save` | Save current conversation |
| `/clear` | Clear conversation history |
| `/quit` or `/exit` | Exit Cortex |
| `/<skill> [request]` | Run a skill on a request (see [Skills](cli.md#skills)) |

---

## OpenTUI Runtime Startup

### OpenTUI sidecar unavailable

If the sidecar frontend binary (or Bun dev runtime) is missing, Cortex exits with an error and prints setup guidance.

Options:

1. Re-run the installer (`curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash`); release wheels include the sidecar. From a source checkout, run `./install.sh` at the repository root.
2. For source development, install Bun and build the sidecar:

```bash
cd frontend/cortex-tui
bun install
bun run build
```

3. Start the worker on its own to see whether it comes up (it waits for JSON-RPC on stdin; Ctrl+C ends it):

```bash
~/.cortex/install/venv/bin/python -P -m cortex --worker-stdio   # installed Cortex
.venv/bin/python -P -m cortex --worker-stdio                     # source checkout
```

If the worker starts but the app still cannot reach it, its two halves are probably from different versions (a source checkout's TUI with an installed worker, for example): reinstall Cortex. `docs/protocol-debugging.md` shows how to talk to the worker by hand.

---

## Installation Issues

### `cortex: command not found`

- **Right after installing:** the installer added `~/.local/bin` to your `PATH` in your shell's profile, which only new terminals read. Open a new terminal, or run the `export PATH=...` line the installer printed. With `CORTEX_NO_MODIFY_PATH=1` it changed no profile: run Cortex by the full path it printed, or add `~/.local/bin` to your `PATH` yourself.
- **It worked before:** check that `~/.local/bin/cortex` still points at an existing file (`ls -lL ~/.local/bin/cortex`). If `~/.cortex/install` was removed, the link is left dangling: run the installer again.

### Python version

The installer needs nothing from you here: it uses a Python 3.11 or later
from your system, or installs a private one when there is none, and a
re-run keeps the one the runtime uses. To change it, run the installer
with `CORTEX_MANAGED_PYTHON=1` (the private one) or `CORTEX_PYTHON=/path/to/python3`
(see the [Installation Guide](installation.md)).

A source checkout (`pip install -e .`) needs Python 3.11 or later
(`python3 --version`). If yours is older:

```bash
python3.12 -m venv .venv      # or install one first, e.g. pyenv install 3.12
source .venv/bin/activate
pip install -e ".[dev]"
```

### Package conflicts

Create a clean virtual environment and reinstall, or rerun the installer (which uses an isolated runtime under `~/.cortex/install`):

```bash
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash
```

---

## Model Selection Issues

### No model loaded

If you see "No model loaded" after starting Cortex, add a provider key and pick a model:

1. `/login openai <api_key>` (or `anthropic` / `azure` / `openai-compatible`), or set the provider's environment variable.
2. `/model` to open the picker, or `/model openai:gpt-5.5` to select directly.

### Model not authenticated

`/model` refuses a provider without a key ("<provider> is not authenticated"). Run `/login <provider> <api_key>` or set the environment variable named in the message, then select the model again.

### Azure or OpenAI-compatible endpoint not configured

Azure needs its resource endpoint, and the `openai-compatible` provider needs its base URL. Give it with the key — `/login azure <api_key> <endpoint>` or `/login openai-compatible <api_key> <base_url>` — and select the model again; no restart is needed. You can instead set `AZURE_OPENAI_ENDPOINT` / `OPENAI_COMPATIBLE_BASE_URL` before starting Cortex, or `cloud_azure_endpoint` / `cloud_openai_compatible_base_url` in `~/.cortex/config.yaml` (read when Cortex starts); either takes precedence over the URL saved by `/login`.

### Model provenance mismatch

Every turn is rejected unless the response proves it came from the selected provider and model. A mismatch usually means a proxy or gateway rewrote the model name, or the endpoint is not the one you configured — check `cloud_openai_compatible_base_url` / `cloud_azure_endpoint` and the model id you selected.

### Reply cut off at the output limit

"The model reached the output limit (max_tokens=N)" means the reply, including the model's reasoning, used the whole output budget. The partial reply stays in the conversation, so you can ask the model to continue; to give it more room, raise `max_tokens` in `~/.cortex/config.yaml`. Some OpenAI-compatible servers (for example a local vLLM with a small context window) reject a `max_tokens` larger than they support; lower it for those endpoints.

### Poor response quality

- Try a stronger model.
- Tune flat keys in `~/.cortex/config.yaml` (`temperature`, `top_p`).

---

## Cloud and Tooling Issues

### Turn reached its limit of model requests

"The turn reached its limit of model requests (tools_max_iterations=N)" means the model kept calling tools until the turn had made N model requests. In an interactive session its tool calls and results stay in the conversation, so a new message (for example "continue") lets it carry on; a headless `-p` run ends with exit code 1. To allow longer turns, raise `tools_max_iterations` in `~/.cortex/config.yaml` (at most 1000).

### Stuck on "Thinking..." for cloud models

1. Valid provider key via `/login <provider>` (or the provider's environment variable)
2. Reasonable timeout/retry values:

```yaml
cloud_timeout_seconds: 60
cloud_max_retries: 2
```

A request that waits longer than `cloud_timeout_seconds` for data fails and, like other connection errors, is sent again up to `cloud_max_retries` times; once a request has been sent again, the status line reads `retry` until the turn ends, and Esc still interrupts the turn. Failures are logged in `~/.cortex/cortex.log` with the provider and model.

### Model emits fake tool JSON or `<tool_calls>` text

When tools are disabled (`tools_enabled: false` or `tools_profile: off`), Cortex injects a no-tools instruction to prevent fake tool-call output. Tools are on by default; if you disabled them and want them back:

```yaml
tools_enabled: true
tools_profile: full
```

### Tool permission prompts

When a tool call needs approval, the TUI shows an arrow menu: **Allow once** / **Allow always** / **Reject** (↑↓ + Enter; Esc rejects). "Allow always" covers only that exact command (run in the same directory) or file, in the directory Cortex was started in. Persistent approvals are stored in `~/.cortex/tool_permissions.yaml` — delete that file to reset them; running sessions pick up the change.

### Headless edits are denied

`cortex -p` denies `edit_file`/`write_file`/`bash` by design. Pass `--full-auto` to auto-approve them.

---

## Configuration Issues

Cortex reads `~/.cortex/config.yaml`; every key is optional (see the [Configuration Guide](configuration.md)). The file is flat (e.g. `temperature`, `max_tokens`) — no nested sections. Edit it with a text editor and restart Cortex; there is no CLI subcommand for configuration.

If Cortex cannot read or write its state files:

```bash
sudo chown -R $(whoami) ~/.cortex/
chmod -R u+rwX ~/.cortex/
```

---

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "No model loaded" | No model selected | `/login <provider> <key>`, then `/model` |
| "No API key configured for <provider>" | Model selected without credentials | `/login <provider> <key>` or the environment variable named in the message |
| "Unsupported provider" | `/model` selector without a known provider | `/model <provider>:<model>` with openai, anthropic, azure, or openai-compatible |
| `[tool] bash refused (needs --full-auto)` in a headless run | Headless runs refuse edits and commands without `--full-auto` | Pass `--full-auto`, or allow chosen commands in `~/.cortex/tool_permissions.yaml` (see the [Configuration Guide](configuration.md)) |
| "Not allowed: a rule in ~/.cortex/tool_permissions.yaml denies this ..." | A deny rule you saved | Edit `~/.cortex/tool_permissions.yaml` |
| "Unknown command" | Typo in slash command | `/help` |

---

## Collecting Diagnostic Information

From inside Cortex: `/status` and `/update`.

From the terminal:

```bash
cortex --version
~/.cortex/install/venv/bin/python --version   # the Python Cortex runs on
uname -a                                      # and sw_vers on macOS
```

Logs are written to `~/.cortex/cortex.log`.

---

## Getting Help

- **Inside Cortex:** `/help` lists all commands.
- **GitHub Issues:** report bugs at https://github.com/faisalmumtaz89/Cortex/issues. Include `cortex --version`, and the end of `~/.cortex/cortex.log` (and `~/.cortex/cortex-tui.log` if it exists).
- **Working on Cortex itself:** see the [Development Guide](development.md).
