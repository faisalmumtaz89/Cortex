# Troubleshooting Guide

## How Cortex Works

Cortex uses a split runtime:
- `cortex` launches an OpenTUI frontend sidecar (terminal renderer)
- frontend talks to Python backend worker (`python -m cortex --worker-stdio`) over JSON-RPC

**Available slash commands:**

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/status` | Show current setup (GPU, model, template) |
| `/gpu` | Show GPU information and memory status |
| `/model` | Manage local/cloud models (load, delete, switch) |
| `/download` | Download a model from HuggingFace |
| `/benchmark` | Run performance benchmark |
| `/template` | Configure chat template for current model |
| `/finetune` | Fine-tune a model interactively |
| `/login` | Manage OpenAI/Anthropic/HuggingFace credentials |
| `/save` | Save current conversation |
| `/clear` | Clear conversation history |
| `/quit` | Exit Cortex |

**Keyboard shortcuts:**

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel current generation, or exit |
| `Ctrl+D` | Exit Cortex |
| `Tab` | Auto-complete commands (readline) |
| `?` | Show in-app help |

---

## OpenTUI Runtime Startup

### OpenTUI sidecar unavailable

If the sidecar frontend binary (or Bun dev runtime) is missing, Cortex exits with an error and prints setup guidance.

Options:

1. For source development, install Bun and frontend deps:
```bash
cd frontend/cortex-tui
npm install
bun run scripts/build.ts
```

2. Verify worker path manually:
```bash
python -m cortex --worker-stdio
```

If worker handshake fails, check protocol version compatibility between frontend and backend (`1.0.0`).

---

## GPU Not Detected

### Cortex requires Apple Silicon

Cortex only runs on Apple Silicon Macs (M1, M2, M3, M4 series) with Metal support. If GPU validation fails at startup, you will see:

```
Error: GPU validation failed. Cortex requires Apple Silicon with Metal support.
```

**Check your hardware:**

```bash
# Verify you are on Apple Silicon (expected output: arm64)
python -c "import platform; print(platform.machine())"

# Check your macOS version
sw_vers
```

**Check GPU details inside Cortex:**

Once Cortex is running, use the `/gpu` command to see detailed GPU information including:
- Chip name (e.g., Apple M1 Pro)
- GPU core count
- Total and available memory
- Metal support status
- MPS (Metal Performance Shaders) support status

Use `/status` for a summary of your current setup including GPU, loaded model, and template.

### Metal support not available

Metal is required for GPU acceleration. Verify Metal support:

```bash
system_profiler SPDisplaysDataType | grep Metal
```

If Metal is not listed, your hardware may not be supported. Cortex requires Metal support; MLX works best on macOS 13.3+.
If you are on Apple Silicon but Metal is missing, update macOS and install Xcode Command Line Tools.

---

## Installation Issues

### MLX framework not available

MLX is the primary inference backend for Cortex on Apple Silicon. If MLX is not installed or fails to load:

```bash
# Reinstall MLX
pip uninstall mlx mlx-lm
pip install "mlx>=0.30.4" "mlx-lm>=0.30.5" --upgrade
```

If MLX compilation fails during installation:

```bash
# Ensure Xcode Command Line Tools are installed
xcode-select --install

# Reinstall with no cache
pip install "mlx>=0.30.4" "mlx-lm>=0.30.5" --verbose --no-cache-dir
```

### PyTorch MPS issues

Cortex can also use PyTorch with MPS (Metal Performance Shaders) backend. If MPS is not available:

```bash
# Check MPS support
python -c "import torch; print(torch.backends.mps.is_available())"

# Update PyTorch
pip install torch torchvision torchaudio --upgrade
```

MPS requires a recent macOS version on Apple Silicon. If `torch.backends.mps.is_available()` is false, upgrade macOS and PyTorch.

### Python version

Cortex requires Python 3.11 or later:

```bash
python --version
```

If your version is older, install a compatible version:

```bash
# Using pyenv
pyenv install 3.11.7
pyenv global 3.11.7

# Or create a fresh virtual environment
python3.11 -m venv cortex-env
source cortex-env/bin/activate
pip install -r requirements.txt
```

### Package conflicts

If you experience import errors or dependency conflicts:

```bash
# Create a clean virtual environment
python -m venv cortex-clean
source cortex-clean/bin/activate
pip install -r requirements.txt
```

---

## Model Loading Issues

### No models found

If you see "No model loaded" after starting Cortex, you need to download or select a model.

Inside the Cortex REPL:

1. Use `/download` to download a model from HuggingFace. You will be prompted for a repository ID (e.g., `meta-llama/Llama-3.2-3B`).
2. Use `/model` to list and load local models that are already downloaded.
3. Or select a cloud model with `/model openai:gpt-5.1` or `/model anthropic:claude-sonnet-4-5`.

For cloud models, configure API keys with `/login openai` or `/login anthropic`.
For gated HuggingFace models (like Llama), use `/login huggingface`.

### Model format not supported

Cortex supports the following model formats:
- **MLX** (recommended for Apple Silicon, best performance)
- **SafeTensors**
- **GGUF**
- **PyTorch**

If a model fails to load, it may be in an unsupported format or may be corrupted. Try downloading it again using `/download`.

### Model too large for available memory

Apple Silicon uses unified memory shared between CPU and GPU. If your model is too large:

- Use a smaller or quantized variant of the model (4-bit or 8-bit quantized models use significantly less memory).
- Check available memory with `/gpu` inside the Cortex REPL.
- Close other memory-intensive applications before loading large models.
- `gpu_memory_fraction` is currently advisory; for large models, prefer MLX conversion or dynamic quantization.

### GPTQ/AWQ models are slow

Some GPTQ/AWQ loaders do not support MPS on macOS and will run on CPU. If performance is poor:

- Prefer MLX models (`mlx-community`) or GGUF for GPU‑accelerated inference
- Use smaller model sizes or MLX conversions

### Previously used model not found

If Cortex reports that a previously used model was not found at startup, the model files may have been moved or deleted. Use `/model` to select a different model or `/download` to re-download it.

---

## Performance Issues

### Slow token generation

Inside the Cortex REPL:

1. Run `/benchmark` to measure tokens per second and first-token latency. The GPU utilization number is a CPU-based proxy and may show 0% for GGUF models.
2. Check `/gpu` to see current memory usage and GPU status.

Common causes of slow generation:
- **Model too large:** If the model barely fits in memory, performance will suffer. Use a smaller or more aggressively quantized model.
- **Background processes:** Other applications consuming GPU memory or compute resources.
- **Thermal throttling:** Extended heavy use may cause the Mac to throttle GPU performance. Allow the machine to cool down.

### High first-token latency

The first token takes longer because the model must process the entire input prompt. This is normal. Longer conversation histories increase this latency.

Use `/clear` to reset the conversation if the context has grown very large and first-token latency is unacceptable.

### GGUF \"skipping kernel\" messages

When loading GGUF models you may see lines like:

```
ggml_metal_init: skipping kernel_xxx_bf16 (not supported)
```

This is expected. It means a BF16-specific kernel is not available on your GPU, so the runtime falls back to the best supported kernel (typically FP16). GPU acceleration is still active.

### Poor response quality

Response quality depends on the model and its configuration:

- Ensure you are using an appropriate chat template. Use `/template` to configure or reconfigure the template for your current model.
- Try a different model. Larger models generally produce better responses but require more memory.
- Generation parameters (temperature, top_p, top_k, repetition_penalty) are configured as flat keys in `config.yaml`. Lower temperature values (e.g., 0.3) produce more focused responses; higher values (e.g., 1.0) produce more varied output.

---

## Configuration Issues

### Config file location

Cortex stores its configuration in `config.yaml`. Default settings are defined in `cortex/config.py`. The YAML file uses a flat key structure (for example `context_length`, `max_tokens`), grouped by topic in the documentation.

Key configuration areas (flat keys):
- **GPU**: Metal backend settings and optimization level
- **Memory**: Unified memory hints
- **Performance**: Batch size and context length
- **Inference**: Temperature, top_p, top_k, repetition penalty, max tokens

### Editing configuration

Edit `config.yaml` directly with a text editor. There is no CLI subcommand for modifying configuration -- you edit the YAML file and restart Cortex for changes to take effect.

Example `config.yaml` adjustments:

```yaml
# Prefer speed‑optimized MLX conversion
gpu_optimization_level: maximum
auto_quantize: true

# Adjust generation parameters
temperature: 0.5
max_tokens: 4096
repetition_penalty: 1.1
```

### Permission issues

If Cortex cannot read or write its configuration or model files:

```bash
# Fix ownership of the Cortex config directory
sudo chown -R $(whoami) ~/.cortex/

# Fix permissions
chmod -R 755 ~/.cortex/
```

---

## Terminal UI Issues

### Display rendering problems

Cortex uses Unicode box-drawing characters and ANSI color codes. If the UI looks garbled:

```bash
# Ensure your terminal supports 256 colors
echo $TERM
# Expected: xterm-256color or similar

# If needed, set the terminal type
export TERM=xterm-256color
```

Use a modern terminal emulator (Terminal.app, iTerm2, Warp, Ghostty) for best results.

### Input issues

Cortex uses a custom input handler with character-by-character reading for its styled input box. If you experience input problems:

- Arrow keys (left/right), Home, End, and Backspace should work within the input box.
- If key input does not work correctly, check that your terminal is not remapping these keys.

---

## Cloud and Tooling Issues

### Stuck on "Thinking..." for cloud models

If cloud generation appears stalled, verify:

1. `cloud_enabled: true` in `config.yaml`
2. Valid provider key via `/login openai` or `/login anthropic`
3. Reasonable timeout/retry values:

```yaml
cloud_timeout_seconds: 60
cloud_max_retries: 2
tools_idle_timeout_seconds: 45
```

Cortex now fails fast on true idle stream timeouts and logs attempt details in `~/.cortex/cortex.log` with provider/model and request id.

### Model emits fake tool JSON or `<tool_calls>` text

When tools are disabled (`tools_enabled: false` or `tools_profile: off`), Cortex injects a no-tools instruction to prevent fake tool-call output.

If you want tools, explicitly enable them:

```yaml
tools_enabled: true
tools_profile: read_only
```

### Tool permission prompts

When tools are enabled, every new permission scope prompts:

- `Allow once`
- `Allow always`
- `Reject`

Pressing `Esc` maps to reject/cancel.
Persistent approvals are stored in `~/.cortex/tool_permissions.yaml`.

---

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "GPU validation failed" | Not running on Apple Silicon or Metal not available | Verify hardware with `python -c "import platform; print(platform.machine())"` |
| "No model loaded" | No model selected | Use `/model` or `/download` in the REPL |
| "Missing API key for openai/anthropic" | Cloud model selected without credentials | Run `/login openai` or `/login anthropic` |
| "Failed to load model" | Model corrupted, wrong format, or insufficient memory | Check `/gpu` for memory, try a smaller model |
| "MLX not available" | MLX not installed or not on Apple Silicon | `pip install "mlx>=0.30.4" "mlx-lm>=0.30.5" --upgrade` |
| "MPS backend not available" | PyTorch MPS not supported | Update PyTorch and verify you are on a recent macOS release |
| "Unknown command" | Typo in slash command | Type `/help` to see available commands |
| "huggingface-hub not installed" | Missing dependency for `/login` | `pip install huggingface-hub` |
| "OpenAI/Anthropic runtime dependency is missing" | Missing provider SDK | Run `/login openai` or `/login anthropic` to auto-install and configure dependencies |

---

## Collecting Diagnostic Information

When reporting issues, gather the following information from inside the Cortex REPL:

1. Run `/status` to get your current setup (GPU, model, template).
2. Run `/gpu` to get detailed GPU and memory information.
3. Run `/benchmark` (if a model is loaded) to get performance metrics.

Also collect system information from the terminal:

```bash
# Hardware info
system_profiler SPHardwareDataType

# GPU info
system_profiler SPDisplaysDataType

# Python and package versions
python --version
pip list | grep -E "mlx|torch|cortex"

# macOS version
sw_vers
```

---

## Getting Help

- **Inside Cortex:** Type `/help` to see all available commands.
- **Keyboard shortcuts:** Type `?` inside the REPL.
- **GitHub Issues:** Report bugs at the project's GitHub Issues page.
- **Source code:** The CLI implementation is in `cortex/ui/cli.py` and the entry point is `cortex/__main__.py`.
