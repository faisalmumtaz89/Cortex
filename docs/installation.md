# Installation Guide

## Supported Systems

| System | Notes |
|---|---|
| macOS 13 or later, Apple Silicon or Intel | |
| Linux x86_64 or arm64 with glibc 2.28 or later | Debian 10, Ubuntu 20.04, RHEL 8, Fedora, Arch, or newer. Not musl distributions such as Alpine. |
| Windows 10 or 11 | Through WSL (Windows Subsystem for Linux); the PowerShell installer sets it up. |

The installer needs only `curl`. It uses a Python 3.11+ from your system when one is there; otherwise it downloads `uv` (checksum-verified) and installs a private Python 3.12 under `~/.cortex/install`, without admin rights or a package manager. Source installs also need Git, and Bun (which the installer installs when missing).

## Recommended Install

**macOS and Linux:**

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

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.ps1 | iex
```

This installs Cortex inside WSL and adds a `cortex` command to Windows that runs it in the current folder. If WSL or a Linux distribution is missing, it says how to add it (`wsl --install`, then restart and open Ubuntu once) and installs nothing. To pin a version, set `$env:CORTEX_VERSION = "X.Y.Z"` first. Projects in your Linux home (`\\wsl$\...`) are much faster than ones under `C:\`.

The installer:

1. Detects your system and processor, and picks the matching wheel (each release has one per supported platform, with the TUI built for it)
2. Uses a system Python 3.11+, or installs a private one with `uv`
3. Creates an isolated runtime under `~/.cortex/install`
4. Downloads the `cortex-llm` wheel from the GitHub release (the single distribution channel), verifies it against the release's `.sha256` asset, and installs the verified wheel — or installs the source checkout when run from a clone
5. Builds the OpenTUI sidecar when missing or stale (installing Bun if needed; release wheels already bundle it)
6. Links the `cortex` launcher into `~/.local/bin` and, when that folder is not on your `PATH`, adds it in your shell's profile (`~/.zshrc`, `~/.bashrc`, fish's `conf.d`, or `~/.profile`); set `CORTEX_NO_MODIFY_PATH=1` to leave profiles alone

A checksum mismatch or a missing `.sha256` asset aborts the install — nothing is installed. If no release has been published yet, the installer says so and points at the source-checkout install instead. Prerelease versions are not installable via `install.sh` — pins are stable `X.Y.Z` only.

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
npm run build      # sidecar binary for this machine's platform
```

## Verification

```bash
cortex
```

Then inside Cortex:

- `/status` — the active model and the model that answered the last turn
- `/login` — which providers have a key configured

## Getting a Model

Add a key for a provider, then pick a model:

```bash
/login openai <api_key>        # or anthropic / azure / openai-compatible
/model openai:gpt-5.5
```

Provider keys can also come from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`); see the [CLI Documentation](cli.md#login--credentials) for Azure and OpenAI-compatible endpoints. Cortex remembers the last selected model across restarts (`~/.cortex/state.yaml`).

## Upgrading

Cortex checks once a day (in the background, never blocking startup) whether a newer Cortex release exists and posts a one-line notice at session start when one does. Opt out with `auto_update_check: false` in `~/.cortex/config.yaml`.

```bash
/update            # inside Cortex: installed vs latest version
/update cortex     # install the latest published Cortex release (restart to apply)
```

Or re-run the installer from a shell:

```bash
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash
```

`/update cortex` only ever installs a pinned, published release that is strictly newer than the running version: it downloads that GitHub release's wheel asset, verifies it against the release's `.sha256` asset, and installs the verified wheel into the running environment. The new version applies when you restart Cortex. Quitting Cortex while the install step is running is safe — the worker finishes that step before exiting rather than interrupting it.

Running from a source checkout (`git clone` + `./install.sh`)? `/update cortex` refuses and points you at `git pull` instead — installing the release wheel would replace your editable install.

## Common Installation Issues

**OpenTUI sidecar missing (source checkout)** — run `./install.sh` at the repository root, or `npm install` in `frontend/cortex-tui`.

**Permission denied on `~/.cortex`**

```bash
sudo chown -R $(whoami) ~/.cortex
chmod -R u+rwX ~/.cortex
```

For everything else, see `docs/troubleshooting.md`.

## Uninstallation

```bash
# Remove the isolated runtime and launcher
rm -rf ~/.cortex/install
rm -f ~/.local/bin/cortex

# Remove configuration, caches, and conversations
rm -rf ~/.cortex
```

API keys saved with `/login` in the system keyring stay there (service
`cortex-llm`); delete them with your keyring tool, such as Keychain Access on
macOS. Keys saved to `~/.cortex/credentials.json` go with `~/.cortex`.

On Windows, also delete `%LOCALAPPDATA%\Programs\Cortex` and remove it from
your user `PATH`.

If you installed into your own environment instead: `pip uninstall cortex-llm`.

## Next Steps

1. Start Cortex: `cortex`
2. Get a model: `/login` + `/model provider:model`
3. Ask for changes to your code — see the [CLI Documentation](cli.md)
4. Tune settings — see the [Configuration Guide](configuration.md)
