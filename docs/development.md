# Development Guide

## Getting Started

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/faisalmumtaz/Cortex.git
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

## Project Structure

```
Cortex/
├── pyproject.toml                  # Build config, dependencies, tool settings
├── config.yaml                     # Default runtime configuration
├── setup.py                        # Legacy setup script
├── requirements.txt                # Pinned dependency list
├── install.py                      # Installation helper script
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
│   ├── ui/                         # User interface
│   │   ├── __init__.py
│   │   ├── cli.py                  # Interactive CLI (CortexCLI)
│   │   ├── markdown_render.py      # Markdown rendering helpers
│   │   └── terminal_app.py         # Textual-based terminal UI
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
├── tests/
│   ├── test_apple_silicon.py       # Apple Silicon GPU tests
│   ├── test_metal_optimization.py  # Metal optimization tests
│   └── verify_gpu_acceleration.py  # GPU acceleration verification
├── docs/                           # Documentation
├── scripts/                        # Helper scripts (reserved)
└── tools/                          # Development/build tools
```

## Application Architecture

The entry point is `cortex/__main__.py`. The `main()` function constructs all components directly -- there is no application class or factory pattern:

```python
config = Config()
gpu_validator = GPUValidator()
model_manager = ModelManager(config, gpu_validator)
inference_engine = InferenceEngine(config, model_manager)
conversation_manager = ConversationManager(config)

cli = CortexCLI(
    config=config,
    gpu_validator=gpu_validator,
    model_manager=model_manager,
    inference_engine=inference_engine,
    conversation_manager=conversation_manager,
)
cli.run()
```

GPU validation runs first. If it fails, the process exits before any other components are created. On shutdown, the inference engine's memory pool is cleaned up and PyTorch MPS caches are flushed.

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
| `CortexCLI` | `cortex.ui.cli` | Interactive command-line interface |
| `TemplateRegistry` | `cortex.template_registry` | Chat template detection and management |
| `FineTuneWizard` | `cortex.fine_tuning` | Interactive fine-tuning workflow |
