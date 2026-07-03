# Installation Guide

## Prerequisites

**Hardware:**

- Apple Silicon Mac (M1–M4 series)
- 8GB+ unified memory (16GB+ recommended for larger local models)
- macOS 13.3+ (MLX minimum requirement)

**Software:**

- Python 3.11+ (3.12 recommended)
- Xcode Command Line Tools
- Git (source installs)
- Bun runtime for frontend source builds (global `bun`, or the local runtime provisioned by `npm install` in `frontend/cortex-tui`; the installer handles this)

**Platform boundary:** local inference requires Apple Silicon (MLX / llama.cpp-Metal). On Linux, Cortex runs **cloud-only from a source install** (`pip install -e .` + `/login`) — no local models, no installer, not officially supported. Windows is not supported natively (POSIX shell tool and TUI build) — use WSL2, which follows the Linux path. Intel Macs are not supported.

## Recommended Install

```bash
# One-line installer
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash

# Pin a specific version
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash -s -- 1.0.18

# Or clone and run locally (installs from the source checkout)
git clone https://github.com/faisalmumtaz89/Cortex.git
cd Cortex
./install.sh

# Start Cortex
cortex
```

The installer:

1. Verifies Apple Silicon and an arm64 Python 3.11+
2. Creates an isolated runtime under `~/.cortex/install`
3. Installs the `cortex-llm` package (or the source checkout)
4. Builds the OpenTUI sidecar when missing or stale (installing Bun if needed)
5. Links the `cortex` launcher into `~/.local/bin`

If `~/.local/bin` is not on your `PATH`, the installer prints the line to add to your shell profile.

## Development Install

```bash
git clone https://github.com/faisalmumtaz89/Cortex.git
cd Cortex
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verify
python -m pytest tests/ -q
```

For the frontend:

```bash
cd frontend/cortex-tui
npm install
npm run typecheck
npm run build      # bundled darwin-arm64 sidecar binary
```

## Verification

```bash
cortex
```

Then inside Cortex:

- `/status` — current setup (GPU, model, settings)
- `/gpu` — GPU information
- `/benchmark` — performance test (after loading a local model)

Verify MLX sees the GPU directly:

```bash
python -c "import mlx.core as mx; print(f'MLX device: {mx.default_device()}')"
```

## Getting a Model

**Local (built-in downloader, recommended):**

```bash
cortex
/download mlx-community/Nanbeige4.1-3B-bf16 --load
```

Prefer `mlx-community` models (native MLX) or GGUF files. Downloads land in `model_path` (default `~/models`).

**Cloud instead of local:**

```bash
/login openai <api_key>        # or /login anthropic <api_key>
/model openai:gpt-5.1
```

**Default model (optional)** — edit `config.yaml` in the directory you run Cortex from:

```yaml
default_model: Nanbeige4.1-3B-bf16
model_path: ~/models
```

Cortex also remembers the last loaded model across restarts (`~/.cortex/state.yaml`).

## Common Installation Issues

**MLX not available**

```bash
# Ensure you are on Apple Silicon
python -c "import platform; print(platform.machine())"   # expect: arm64

pip uninstall mlx mlx-lm
pip install "mlx>=0.30.4" "mlx-lm>=0.30.5"
```

**OpenTUI sidecar missing (source checkout)** — run `./install.sh` at the repository root, or `npm install` in `frontend/cortex-tui`.

**Permission denied on `~/.cortex`**

```bash
sudo chown -R $(whoami) ~/.cortex
chmod -R 755 ~/.cortex
```

For everything else, see `docs/troubleshooting.md`.

## Uninstallation

```bash
# Remove the isolated runtime and launcher
rm -rf ~/.cortex/install
rm -f ~/.local/bin/cortex

# Remove configuration, caches, and conversations
rm -rf ~/.cortex

# Remove downloaded models (optional)
rm -rf ~/models
```

If you installed into your own environment instead: `pip uninstall cortex-llm`.

## Next Steps

1. Start Cortex: `cortex`
2. Get a model: `/download ... --load` or `/login` + `/model provider:model`
3. Ask for changes to your code — see the [CLI Documentation](cli.md)
4. Tune settings — see the [Configuration Guide](configuration.md)
