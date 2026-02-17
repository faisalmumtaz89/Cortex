# Development Guide

## Getting Started

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/faisalmumtaz89/Cortex.git
cd Cortex

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Prerequisites

- macOS on Apple Silicon (ARM64)
- Python 3.11 or later
- Bun runtime (global `bun` or local runtime from `frontend/cortex-tui/npm install`)

## Project Structure

```
Cortex/
├── pyproject.toml                  # Build config, dependencies, tool settings
├── config.yaml                     # Default runtime configuration
├── setup.py                        # Packaging/install shim
├── requirements.txt                # Pinned dependency list
├── install.sh                      # Shell-based installer
├── cortex/
│   ├── __init__.py                 # Package init, system requirement checks
│   ├── __main__.py                 # Entry point - wires all components together
│   ├── config.py                   # Configuration loading (Config class)
│   ├── gpu_validator.py            # GPU detection and validation (GPUValidator)
│   ├── model_manager.py            # Model loading and management (ModelManager)
│   ├── model_downloader.py         # HuggingFace model downloads (ModelDownloader)
│   ├── inference_engine.py         # Text generation (InferenceEngine)
│   ├── conversation_manager.py     # Chat history and persistence (ConversationManager)
│   ├── app/                        # Worker application service layer
│   │   ├── worker_runtime.py       # JSON-RPC runtime assembly
│   │   ├── session_service.py      # Session orchestration
│   │   ├── model_service.py        # Model/auth operations
│   │   └── permission_service.py   # Permission ask/reply bridge
│   ├── protocol/                   # JSON-RPC protocol contracts/server
│   │   ├── rpc_server.py
│   │   ├── schema.py
│   │   └── types.py
│   ├── metal/                      # GPU acceleration layer
│   │   ├── __init__.py
│   │   ├── gpu_validator.py        # Metal-level GPU validation
│   │   ├── memory_pool.py          # GPU memory pool management
│   │   ├── optimizer.py            # Unified MLX/MPS optimizer
│   │   ├── mps_optimizer.py        # PyTorch MPS backend optimizer
│   │   ├── mlx_accelerator.py      # MLX framework integration
│   │   ├── mlx_compat.py           # MLX compatibility patches
│   │   ├── mlx_converter.py        # Model conversion for MLX
│   │   └── performance_profiler.py # GPU performance profiling
│   ├── ui/                         # Terminal UI helpers
│   │   ├── __init__.py
│   │   ├── cli_commands.py         # Slash command dispatch helpers
│   │   └── generation.py           # Streaming response helpers
│   ├── ui_runtime/                 # OpenTUI launcher + bundled binary
│   │   ├── launcher.py
│   │   └── bin/cortex-tui
│   ├── fine_tuning/                # Fine-tuning support
│   │   ├── __init__.py
│   │   ├── wizard.py               # Interactive fine-tuning wizard
│   │   ├── trainer.py              # Training loop
│   │   ├── mlx_lora_trainer.py     # MLX LoRA training
│   │   └── dataset.py              # Dataset handling
│   ├── quantization/               # Model quantization
│   │   ├── __init__.py
│   │   └── dynamic_quantizer.py    # Dynamic quantization
│   ├── template_registry/          # Chat template management
│   │   ├── __init__.py
│   │   ├── registry.py             # Template registry
│   │   ├── auto_detector.py        # Automatic template detection
│   │   ├── config_manager.py       # Template configuration
│   │   ├── interactive.py          # Interactive template setup
│   │   └── template_profiles/      # Built-in template profiles
│   └── utils/
│       └── param_utils.py          # Parameter utilities
├── frontend/cortex-tui/            # OpenTUI + Solid frontend source
├── tests/
│   ├── test_apple_silicon.py       # Apple Silicon GPU tests
│   ├── test_metal_optimization.py  # Metal optimization tests
│   └── verify_gpu_acceleration.py  # GPU acceleration verification
├── docs/                           # Documentation
├── scripts/                        # Helper scripts (reserved)
└── tools/                          # Development/build tools
```

## Application Architecture

Runtime split:
- Frontend: OpenTUI sidecar (`frontend/cortex-tui`)
- Backend: Python worker (`python -m cortex --worker-stdio`)

The entry point is `cortex/__main__.py`. By default it launches the OpenTUI sidecar. Worker mode constructs backend services and starts JSON-RPC stdio server:

```python
python -m cortex --worker-stdio
```

GPU validation runs first. If it fails, the process exits before any other components are created. On shutdown, the inference engine's memory pool is cleaned up and PyTorch MPS caches are flushed.

Frontend source commands:

```bash
cd frontend/cortex-tui
npm install
npm run typecheck
npm run dev        # Runs OpenTUI from source
npm run build      # Builds darwin-arm64 bundled sidecar binary
```

## Running Tests

The test suite lives in the `tests/` directory:

```bash
# Run all tests
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_apple_silicon.py
python -m pytest tests/test_metal_optimization.py

# Run with verbose output
python -m pytest tests/ -v

# Coverage reporting requires pytest-cov (not installed by default)
# python -m pytest tests/ --cov=cortex
```

## Code Style

Formatting and linting tools are configured in `pyproject.toml`:

- **Black** -- code formatter (line length 100)
- **Ruff** -- linter (line length 100, rules: E, F, I, N, W)
- **mypy** -- type checker

```bash
# Format code
black cortex/ tests/

# Lint
ruff check cortex/ tests/

# Type check
mypy cortex/
```

The relevant `pyproject.toml` sections:

```toml
[tool.black]
line-length = 100
target-version = ['py311', 'py312']

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
```

## Key Classes Reference

| Class | Module | Purpose |
|---|---|---|
| `Config` | `cortex.config` | Loads and holds runtime configuration |
| `GPUValidator` | `cortex.gpu_validator` | Validates Apple Silicon GPU availability |
| `ModelManager` | `cortex.model_manager` | Loads, manages, and serves models |
| `ModelDownloader` | `cortex.model_downloader` | Downloads models from HuggingFace Hub |
| `InferenceEngine` | `cortex.inference_engine` | Runs text generation with `GenerationRequest` |
| `ConversationManager` | `cortex.conversation_manager` | Manages chat history and persistence |
| `WorkerRuntime` | `cortex.app.worker_runtime` | JSON-RPC session runtime for OpenTUI |
| `TemplateRegistry` | `cortex.template_registry` | Chat template detection and management |
| `FineTuneWizard` | `cortex.fine_tuning` | Interactive fine-tuning workflow |
