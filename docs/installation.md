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

**Not supported:** Intel Macs, Linux, Windows. Cortex runs on Apple Silicon Macs only.

## Recommended Install

```bash
# One-line installer (latest GitHub release)
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash

# Pin a release version X.Y.Z (installs the vX.Y.Z GitHub release's wheel)
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash -s -- X.Y.Z

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
3. Downloads the `cortex-llm` wheel from the GitHub release (the single distribution channel), verifies it against the release's `.sha256` asset, and installs the verified wheel — or installs the source checkout when run from a clone
4. Builds the OpenTUI sidecar when missing or stale (installing Bun if needed; release wheels already bundle it)
5. Links the `cortex` launcher into `~/.local/bin`

A checksum mismatch or a missing `.sha256` asset aborts the install — nothing is installed. If no release has been published yet, the installer says so and points at the source-checkout install instead. Prerelease versions are not installable via `install.sh` — pins are stable `X.Y.Z` only.

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
/download qwen3-5-9b:q4_0
```

Local models are downloaded and managed by the Lumen engine (`/model` lists what's supported; the cache lives at `~/Library/Caches/lumen/` on macOS).

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

## Upgrading

Cortex checks once a day (in the background, never blocking startup) whether a newer Cortex or Lumen release exists and posts a one-line notice at session start when one does. Opt out with `auto_update_check: false` in `config.yaml`.

**Cortex:**

```bash
/update            # inside Cortex: installed vs latest for Cortex and Lumen
/update cortex     # install the latest published Cortex release (restart to apply)
```

Or re-run the installer from a shell:

```bash
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash
```

`/update cortex` only ever installs a pinned, published release that is strictly newer than the running version: it downloads that GitHub release's wheel asset, verifies it against the release's `.sha256` asset, and installs the verified wheel into the running environment. The new version applies when you restart Cortex. Quitting Cortex while the install step is running is safe — the worker finishes that step before exiting rather than interrupting it.

Running from a source checkout (`git clone` + `./install.sh`)? `/update cortex` refuses and points you at `git pull` instead — installing the release wheel would replace your editable install.

**Lumen (local inference engine):**

```bash
/update lumen      # inside Cortex: stops the managed server, upgrades, verifies
```

Because the update stops the managed `lumen-server`, `/update lumen` refuses to start while a turn, model load, or download is in progress — finish (or interrupt) the current work first.

Or re-run the Lumen installer from a shell (stop Cortex first so no old server keeps running):

```bash
curl -fsSL https://servelumen.com/install.sh | bash
```

After `/update lumen` the managed `lumen-server` restarts automatically the next time a local model is used.

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
