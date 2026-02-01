# Installation Guide

## Quick Start

### Prerequisites

**Hardware Requirements:**
- Apple Silicon Mac (M1-M4 series)
- 8GB+ unified memory (16GB+ recommended)
- macOS 13.3+ (MLX minimum requirement)

**Software Requirements:**
- Python 3.11+ (3.12 recommended)
- Xcode Command Line Tools
- Git

**Not Supported:**
- Intel Macs
- Linux/Windows

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/faisalmumtaz/Cortex.git
cd Cortex

# Run the installer (recommended)
./install.sh

# OR manually install dependencies
pip install -r requirements.txt
pip install -e .

# Start Cortex
cortex
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests to verify installation
python -m pytest tests/
```

## Detailed Installation

### 1. System Dependencies

**Install Xcode Command Line Tools:**
```bash
xcode-select --install
```

**Install Homebrew (if not installed):**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Install Python 3.11+ via Homebrew:**
```bash
brew install python@3.11
# Or use pyenv for version management
brew install pyenv
pyenv install 3.11.7
pyenv global 3.11.7
```

### 2. Virtual Environment Setup

**Using venv (recommended):**
```bash
python3.11 -m venv cortex-env
source cortex-env/bin/activate  # macOS
```

**Using conda:**
```bash
conda create -n cortex python=3.11
conda activate cortex
```

### 3. GPU Dependencies

**Install MLX Framework:**
```bash
pip install "mlx>=0.30.4" "mlx-lm>=0.30.5"
```

**Install PyTorch with MPS support:**
```bash
pip install torch torchvision torchaudio
```

**Verify GPU Support:**
```bash
python -c "import mlx.core as mx; print(f'MLX device: {mx.default_device()}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 4. Install Cortex

**From PyPI (recommended):**
```bash
pip install cortex-llm
```

**From PyPI with pipx (isolated install):**
```bash
pipx install cortex-llm
```

**From Source (development):**
```bash
git clone https://github.com/faisalmumtaz/Cortex.git
cd Cortex
pip install -e .
```

### 5. Configuration Setup

Configuration is automatically set up during installation. You can modify settings by editing `config.yaml` in the project root:

```bash
# Start Cortex and use /status to see current configuration
cortex
# Then type: /status
```

## Verification

### Test Basic Functionality

```bash
# Start Cortex interactive mode
cortex

# Then use these commands inside Cortex:
# /status      - Check current setup
# /gpu         - Show GPU information  
# /benchmark   - Run performance test (if model loaded)
```

### Expected Output

When you run `cortex` and then `/status`, you should see:
```
╭─ Current Setup ──────────────────────────────────────╮
│                                                      │
│  GPU: Apple Silicon M3 Pro                          │
│  Cores: 18                                          │ 
│  Memory: 36.0 GB                                    │
│  Model: None loaded                                  │
│                                                      │
╰──────────────────────────────────────────────────────╯
```

## Model Installation

### Downloading Models

**Using Cortex's built-in downloader (recommended):**
```bash
# Start Cortex
cortex

# Then use the download command
/download

# Or specify a model directly
/download microsoft/DialoGPT-medium
```

**Manual Model Installation:**
```bash
# Create models directory
mkdir -p ~/models

# Download from Hugging Face (example)
git lfs clone https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.2-4bit ~/models/mistral-7b-4bit
```

### Model Configuration

**Set default model:**
Edit the `config.yaml` file in the project root:
```yaml
default_model: 'mistral-7b-instruct'
model_path: ~/models
```

Or set it through the last used model (Cortex remembers your last loaded model):
```bash
cortex
# Load a model with: /model path/to/your/model
# Cortex will remember this as the last used model
```

## Troubleshooting

### Common Installation Issues

**Issue: MLX not available**
```bash
# Solution: Ensure you're on Apple Silicon
python -c "import platform; print(platform.machine())"
# Should output: arm64

# Reinstall MLX
pip uninstall mlx mlx-lm
pip install "mlx>=0.30.4" "mlx-lm>=0.30.5"
```

**Issue: MPS backend not available**
```bash
# Solution: Update PyTorch
pip install torch torchvision torchaudio --upgrade

# Check macOS version
sw_vers
# Requires a recent macOS version for MLX/MPS
```

**Issue: Memory errors**
```bash
# Solution: Use MLX conversion or dynamic quantization
# Edit config.yaml:
auto_quantize: true
gpu_optimization_level: maximum
```

**Issue: Permission denied**
```bash
# Solution: Fix permissions
sudo chown -R $(whoami) ~/.cortex
chmod -R 755 ~/.cortex
```

### Performance Issues

**Slow inference:**
```bash
# Check GPU utilization within Cortex:
cortex
/gpu           # Check GPU status
/benchmark     # Run performance test

# Enable optimizations in config.yaml:
gpu_optimization_level: maximum
```

**High memory usage:**
```bash
# Enable memory optimization in config.yaml:
default_quantization: Q4_K_M
auto_quantize: true
```

### Diagnostic Commands

Cortex provides built-in diagnostics through its interactive interface:

```bash
# Start Cortex
cortex

# Then use these diagnostic commands:
/status        # Current system status
/gpu          # GPU information
/benchmark    # Performance test (with loaded model)
```

For detailed system information, you can run:
```bash
# Check system info
system_profiler SPHardwareDataType
system_profiler SPDisplaysDataType

# Run comprehensive validation
python tests/test_apple_silicon.py
```

## Uninstallation

### Remove Cortex

```bash
# Uninstall package
pip uninstall cortex-llm

# Remove configuration and cache
rm -rf ~/.cortex

# Remove models (optional)
rm -rf ~/models
```

### Clean Installation

```bash
# Remove all Python packages
pip freeze | xargs pip uninstall -y

# Reinstall from scratch
pip install cortex-llm
```

## Advanced Installation

### Build from Source

```bash
# Install build dependencies
pip install build wheel

# Build package
python -m build

# Install built package
pip install dist/cortex_llm-*.whl
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/faisalmumtaz/Cortex.git

# Install development dependencies
pip install -e ".[dev]"

# Run full test suite
python -m pytest tests/
```

## Platform-Specific Notes

### macOS (Apple Silicon)

- **Recommended:** macOS 13.3+ for MLX support
- **Best performance:** Newer macOS releases with up‑to‑date Metal drivers
- **Memory:** Unified memory shared between CPU and GPU
- **Performance:** Best with higher-end Apple Silicon chips and high memory configurations

### Other Platforms

Cortex is not supported on Intel Macs, Linux, or Windows.

## Next Steps

After successful installation:

1. **Start Cortex:** `cortex`
2. **Download a model:** Use `/download` inside Cortex
3. **Start chatting:** Type your message after downloading a model
4. **Get help:** Use `/help` for available commands

For detailed usage instructions, see the [CLI Documentation](cli.md) and [Configuration Guide](configuration.md).
