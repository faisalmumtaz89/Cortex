# Troubleshooting Guide

## How Cortex Works

Cortex uses a split runtime:

- `cortex` launches an OpenTUI frontend sidecar (terminal renderer)
- the frontend talks to the Python backend worker (`python -m cortex --worker-stdio`) over JSON-RPC
- `cortex -p "..."` runs one headless agent turn through the same worker wiring

**Available slash commands:**

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/status` | Show current setup (GPU, model, settings) |
| `/gpu` | Show GPU information and memory status |
| `/model` | List/switch local and cloud models |
| `/download` | Download a model from HuggingFace |
| `/setup` | Load the first local model if none is active |
| `/benchmark` | Run performance benchmark (local models) |
| `/template` | Configure chat template for the current local model |
| `/login` | Manage OpenAI/Anthropic/HuggingFace credentials |
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
python -m cortex --worker-stdio
```

If the worker handshake fails, check protocol version compatibility between frontend and backend (`1.0.0`). See `docs/protocol-debugging.md`.

---

## GPU Not Detected

Cortex only runs on Apple Silicon Macs (M1, M2, M3, M4 series) with Metal support and the MLX framework. If GPU validation fails, local models are unavailable (cloud models may still work in worker/headless mode).

**Check your hardware:**

```bash
# Verify you are on Apple Silicon (expected output: arm64)
python -c "import platform; print(platform.machine())"

# Verify Metal support
system_profiler SPDisplaysDataType | grep Metal

# Check your macOS version (13.3+ recommended for MLX)
sw_vers
```

Inside Cortex, `/gpu` shows chip name, GPU cores, memory, and Metal/MLX support status; `/status` summarizes the current setup.

---

## Installation Issues

### MLX framework not available

MLX is the primary inference backend. If MLX is not installed or fails to load:

```bash
pip uninstall mlx mlx-lm
pip install "mlx>=0.30.4" "mlx-lm>=0.30.5" --upgrade
```

If MLX compilation fails during installation:

```bash
xcode-select --install
pip install "mlx>=0.30.4" "mlx-lm>=0.30.5" --verbose --no-cache-dir
```

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

## Model Loading Issues

### No models found

If you see "No model loaded" after starting Cortex:

1. `/download mlx-community/Nanbeige4.1-3B-bf16 --load` to fetch and load a local model.
2. `/model` to list and load models that are already downloaded.
3. Or select a cloud model: `/model openai:gpt-5.1` or `/model anthropic:claude-sonnet-4-5` (configure keys with `/login`).

For gated HuggingFace models (like Llama), run `huggingface-cli login` in your shell and check with `/login huggingface`.

### Model format not supported

Cortex loads **MLX** and **GGUF** models only. Plain HuggingFace safetensors models are converted to MLX automatically on load. Bare PyTorch checkpoints and GPTQ/AWQ-quantized models are not supported — download an `mlx-community` variant or a GGUF file instead.

### Model too large for available memory

Apple Silicon uses unified memory shared between CPU and GPU. If your model is too large:

- Use a smaller or more aggressively quantized variant (4-bit MLX or Q4 GGUF).
- Check available memory with `/gpu`.
- Close other memory-intensive applications before loading.

### Previously used model not found

If Cortex reports that a previously used model was not found at startup, the files were moved or deleted. Use `/model` to select another model or `/download` to re-fetch it.

---

## Performance Issues

### Slow token generation

1. Run `/benchmark` to measure tokens/second and first-token latency. The GPU utilization number is a CPU-based proxy and may show 0% for GGUF models.
2. Check `/gpu` for current memory usage.

Common causes: model barely fits in memory, background processes consuming resources, or thermal throttling.

### High first-token latency

The first token takes longer because the model must process the entire prompt; long conversations increase this. Use `/clear` to reset the context if it has grown very large.

### GGUF "skipping kernel" messages

Lines like `ggml_metal_init: skipping kernel_xxx_bf16 (not supported)` are expected — a BF16 kernel is unavailable on your GPU and the runtime falls back to FP16. GPU acceleration is still active.

### Poor response quality

- Ensure the right chat template with `/template` (local models).
- Try a larger model, or a cloud model for harder tasks.
- Tune flat keys in `config.yaml` (`temperature`, `top_p`, `repetition_penalty`).

---

## Cloud and Tooling Issues

### Stuck on "Thinking..." for cloud models

1. `cloud_enabled: true` in `config.yaml`
2. Valid provider key via `/login openai` or `/login anthropic`
3. Reasonable timeout/retry values:

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

Cortex reads `config.yaml` from the directory it starts in; defaults live in `cortex/config.py`. The file is flat (e.g. `context_length`, `max_tokens`) — no nested sections. Edit it with a text editor and restart Cortex; there is no CLI subcommand for configuration.

If Cortex cannot read or write its state files:

```bash
sudo chown -R $(whoami) ~/.cortex/
chmod -R 755 ~/.cortex/
```

---

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "GPU validation failed" | Not on Apple Silicon or Metal/MLX missing | Verify with `python -c "import platform; print(platform.machine())"` |
| "No model loaded" | No model selected | `/model`, `/setup`, or `/download ... --load` |
| "Missing API key for openai/anthropic" | Cloud model selected without credentials | `/login openai <key>` or `/login anthropic <key>` |
| "Failed to load model" | Corrupted, unsupported format, or insufficient memory | Check `/gpu`, use an MLX or GGUF variant |
| "Unsupported model format" | PyTorch/GPTQ/AWQ model | Use an `mlx-community` or GGUF variant |
| "MLX not available" | MLX not installed or not on Apple Silicon | `pip install "mlx>=0.30.4" "mlx-lm>=0.30.5" --upgrade` |
| "Permission denied by rule" / rejected tool calls | Headless without `--full-auto`, or a persisted deny rule | Pass `--full-auto`, or edit `~/.cortex/tool_permissions.yaml` |
| "Unknown command" | Typo in slash command | `/help` |
| "huggingface-hub not installed" | Missing dependency for `/login huggingface` | `pip install huggingface-hub` |

---

## Collecting Diagnostic Information

From inside Cortex: `/status`, `/gpu`, and `/benchmark` (with a local model loaded).

From the terminal:

```bash
system_profiler SPHardwareDataType
system_profiler SPDisplaysDataType
python --version
pip list | grep -E "mlx|llama|cortex"
sw_vers
```

Logs are written to `~/.cortex/cortex.log`.

---

## Getting Help

- **Inside Cortex:** `/help` lists all commands.
- **GitHub Issues:** report bugs at the project's GitHub Issues page.
- **Source code:** the entry point is `cortex/__main__.py`; the worker runtime is `cortex/app/worker_runtime.py`.
