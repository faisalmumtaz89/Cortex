# Cortex

GPU-accelerated local LLMs on Apple Silicon, built for the terminal.

Cortex is a fast, native CLI for running and fine-tuning LLMs on Apple Silicon using MLX and Metal. It automatically detects chat templates, supports multiple model formats, and keeps your workflow inside the terminal.

## Highlights

- Apple Silicon GPU acceleration via MLX (primary) and PyTorch MPS
- Multi-format model support: MLX, GGUF, SafeTensors, PyTorch, GPTQ, AWQ
- Built-in LoRA fine-tuning wizard
- Chat template auto-detection (ChatML, Llama, Alpaca, Gemma, Reasoning)
- Conversation history with autosave and export

## Quick Start

```bash
pipx install cortex-llm
cortex
```

Inside Cortex:

- `/download` to fetch a model from HuggingFace
- `/model` to load or manage models
- `/status` to confirm GPU acceleration and current settings

## Installation

### Option A: pipx (recommended)

```bash
pipx install cortex-llm
```

### Option B: from source

```bash
git clone https://github.com/faisalmumtaz/Cortex.git
cd Cortex
./install.sh
```

The installer checks Apple Silicon compatibility, creates a venv, installs dependencies from `pyproject.toml`, and sets up the `cortex` command.

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

## Configuration

Cortex reads `config.yaml` from the current working directory. For tuning GPU memory limits, quantization defaults, and inference parameters, see:

- `docs/configuration.md`

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

## Contributing

Contributions are welcome. See `docs/development.md` for setup and workflow.

## License

MIT License. See `LICENSE`.

---

Note: Cortex requires Apple Silicon. Intel Macs are not supported.
