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
| `/update` | Show installed vs latest version, or update Cortex |
| `/save` | Save current conversation |
| `/clear` | Clear conversation history |
| `/quit` | Exit Cortex |

---

## OpenTUI Runtime Startup

### OpenTUI sidecar unavailable

If the sidecar frontend binary (or Bun dev runtime) is missing, Cortex exits with an error and prints setup guidance.

Options:

1. Run `./install.sh` at the repository root to build and install the sidecar.
2. For source development, install Bun and frontend deps:

```bash
cd frontend/cortex-tui
npm install
```

3. Verify the worker path manually:

```bash
python -P -m cortex --worker-stdio
```

If the worker handshake fails, check protocol version compatibility between frontend and backend (`1.0.0`). See `docs/protocol-debugging.md`.

---

## Installation Issues

### Python version

Cortex requires Python 3.11 or later (`python --version`). If your version is older:

```bash
# Using pyenv
pyenv install 3.12
pyenv global 3.12

# Or a fresh virtual environment with a newer interpreter
python3.12 -m venv cortex-env
source cortex-env/bin/activate
pip install -e .
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
2. `/model` to open the picker, or `/model openai:gpt-5.1` to select directly.

### Model not authenticated

`/model` refuses a provider without a key ("<provider> is not authenticated"). Run `/login <provider> <api_key>` or set the environment variable named in the message, then select the model again.

### Azure or OpenAI-compatible endpoint not configured

Azure needs its resource endpoint (`AZURE_OPENAI_ENDPOINT` or `cloud_azure_endpoint`), and the `openai-compatible` provider needs its base URL (`OPENAI_COMPATIBLE_BASE_URL` or `cloud_openai_compatible_base_url`). Set the value in the environment or `~/.cortex/config.yaml`, then select the model again.

### Model provenance mismatch

Every turn is rejected unless the response proves it came from the selected provider and model. A mismatch usually means a proxy or gateway rewrote the model name, or the endpoint is not the one you configured — check `cloud_openai_compatible_base_url` / `cloud_azure_endpoint` and the model id you selected.

### Reply cut off at the output limit

"The model reached the output limit (max_tokens=N)" means the reply, including the model's reasoning, used the whole output budget. The partial reply stays in the conversation, so you can ask the model to continue; to give it more room, raise `max_tokens` in `~/.cortex/config.yaml`. Some OpenAI-compatible servers (for example a local vLLM with a small context window) reject a `max_tokens` larger than they support; lower it for those endpoints.

### Poor response quality

- Try a stronger model.
- Tune flat keys in `~/.cortex/config.yaml` (`temperature`, `top_p`).

---

## Cloud and Tooling Issues

### Stuck on "Thinking..." for cloud models

1. Valid provider key via `/login <provider>` (or the provider's environment variable)
2. Reasonable timeout/retry values:

```yaml
cloud_timeout_seconds: 60
cloud_max_retries: 2
tools_idle_timeout_seconds: 45
```

Cortex fails fast on true idle stream timeouts and logs attempt details in `~/.cortex/cortex.log` with provider/model and request id.

### Model emits fake tool JSON or `<tool_calls>` text

When tools are disabled (`tools_enabled: false` or `tools_profile: off`), Cortex injects a no-tools instruction to prevent fake tool-call output. Tools are on by default; if you disabled them and want them back:

```yaml
tools_enabled: true
tools_profile: full
```

### Tool permission prompts

When a tool call needs approval, the TUI shows an arrow menu: **Allow once** / **Allow always** / **Reject** (↑↓ + Enter; Esc rejects). Persistent approvals are stored in `~/.cortex/tool_permissions.yaml` — delete that file to reset them.

### Headless edits are denied

`cortex -p` denies `edit_file`/`write_file`/`bash` by design. Pass `--full-auto` to auto-approve them.

---

## Configuration Issues

Cortex reads `~/.cortex/config.yaml`; defaults live in `cortex/config.py`. The file is flat (e.g. `temperature`, `max_tokens`) — no nested sections. Edit it with a text editor and restart Cortex; there is no CLI subcommand for configuration.

If Cortex cannot read or write its state files:

```bash
sudo chown -R $(whoami) ~/.cortex/
chmod -R 755 ~/.cortex/
```

---

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "No model loaded" | No model selected | `/login <provider> <key>`, then `/model` |
| "No API key configured for <provider>" | Model selected without credentials | `/login <provider> <key>` or the environment variable named in the message |
| "Unsupported provider" | `/model` selector without a known provider | `/model <provider>:<model>` with openai, anthropic, azure, or openai-compatible |
| "Permission denied by rule" / rejected tool calls | Headless without `--full-auto`, or a persisted deny rule | Pass `--full-auto`, or edit `~/.cortex/tool_permissions.yaml` |
| "Unknown command" | Typo in slash command | `/help` |

---

## Collecting Diagnostic Information

From inside Cortex: `/status` and `/update`.

From the terminal:

```bash
cortex --version
python --version
sw_vers
```

Logs are written to `~/.cortex/cortex.log`.

---

## Getting Help

- **Inside Cortex:** `/help` lists all commands.
- **GitHub Issues:** report bugs at the project's GitHub Issues page.
- **Source code:** the entry point is `cortex/__main__.py`; the worker runtime is `cortex/app/worker_runtime.py`.
