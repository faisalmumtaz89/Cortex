# Cortex

GPU-accelerated local LLMs on Apple Silicon, built for the terminal.

![Cortex preview](docs/assets/cortex.gif)

Cortex is a fast, native CLI for running and fine-tuning LLMs on Apple Silicon using MLX and Metal. It automatically detects chat templates, supports multiple model formats, and keeps your workflow inside the terminal.

Runtime architecture:
- OpenTUI frontend sidecar for terminal rendering
- Python worker for inference/tools over JSON-RPC (`python -m cortex --worker-stdio`)

## Highlights

- Apple Silicon GPU acceleration via MLX (primary) and PyTorch MPS
- Cloud API models via OpenAI and Anthropic (API key auth)
- Event-driven tool runtime with explicit permission prompts (opt-in)
- Multi-format model support: MLX, GGUF, SafeTensors, PyTorch, GPTQ, AWQ
- Built-in LoRA fine-tuning wizard
- Chat template auto-detection (ChatML, Llama, Alpaca, Gemma, Reasoning)
- Conversation history with autosave and export

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash
cortex
```

If OpenTUI sidecar is unavailable in a source checkout:

```bash
cd frontend/cortex-tui
npm install
```

Inside Cortex:

- `/download` to fetch a model from HuggingFace
- `/login openai` or `/login anthropic` to add cloud API credentials
- `/model` to switch between local and cloud models
- `/status` to confirm GPU acceleration and current settings

## Installation

### Option A: one-line installer (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash
```

### Option B: from source

```bash
git clone https://github.com/faisalmumtaz89/Cortex.git
cd Cortex
./install.sh
```

The installer checks Apple Silicon compatibility, provisions an isolated runtime under `~/.cortex/install`, installs Cortex, builds the OpenTUI sidecar when missing or stale, and sets up the `cortex` command launcher.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13.3+
- Python 3.11+
- 16GB+ unified memory (24GB+ recommended for larger models)
- Xcode Command Line Tools

## Model Support

Cortex supports:

- **MLX** (recommended)
- **GGUF** (llama.cpp + Metal)
- **SafeTensors**
- **PyTorch** (Transformers + MPS)
- **GPTQ** / **AWQ** quantized models

## Advanced Features

- **Dynamic quantization fallback** for PyTorch/SafeTensors models that do not fit GPU memory (INT8 preferred, INT4 fallback)
  - `docs/dynamic-quantization.md`
- **MLX conversion with quantization recipes** (4/5/8-bit, mixed precision) for speed vs quality control
  - `docs/mlx-acceleration.md`
- **LoRA fine-tuning wizard** for local adapters (`/finetune`)
  - `docs/fine-tuning.md`
- **Template registry and auto-detection** for chat formatting (ChatML, Llama, Alpaca, Gemma, Reasoning)
  - `docs/template-registry.md`
- **Inference engine details** and backend behavior
  - `docs/inference-engine.md`
- **CLI workflow and command reference**
  - `docs/cli.md`

## Configuration

Cortex reads `config.yaml` from the current working directory. For tuning GPU memory limits, quantization defaults, and inference parameters, see:

- `docs/configuration.md`

### Tooling Runtime (Opt-In)

Tools are disabled by default for conservative rollout:

```yaml
tools_enabled: false
tools_profile: off
tools_local_mode: disabled
```

To enable read-only cloud tools:

```yaml
tools_enabled: true
tools_profile: read_only
```

Profiles:

- `off`: no tools exposed to models.
- `read_only`: `list_dir`, `read_file`, `search`.
- `patch`: read-only + `apply_patch`.
- `full`: patch profile + `bash`.

Permissions are prompted at runtime with `Allow once`, `Allow always`, or `Reject`.
Persistent approvals are stored in `~/.cortex/tool_permissions.yaml`.

## Documentation

Start here:

- `docs/installation.md`
- `docs/cli.md`
- `docs/model-management.md`
- `docs/troubleshooting.md`

Advanced topics:

- `docs/mlx-acceleration.md`
- `docs/inference-engine.md`
- `docs/dynamic-quantization.md`
- `docs/template-registry.md`
- `docs/fine-tuning.md`
- `docs/development.md`
- `docs/architecture-runtime.md`

## Contributing

Contributions are welcome. See `docs/development.md` for setup and workflow.

## License

MIT License. See `LICENSE`.

---

Note: Cortex requires Apple Silicon. Intel Macs are not supported.
