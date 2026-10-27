# Installation Guide

## Supported Systems

| System | Notes |
|---|---|
| macOS 13 or later, Apple Silicon or Intel | |
| Linux x86_64 or arm64 with glibc 2.28 or later | Debian 10, Ubuntu 20.04, RHEL 8, Fedora, Arch, or newer. Not musl distributions such as Alpine. |
| Windows 10 or 11 | Through WSL (Windows Subsystem for Linux); the PowerShell installer sets it up. |

The installer needs only `curl`. It uses a Python 3.11+ from your system when one is there (the first of `python3.13`, `python3.12`, `python3.11` and `python3` that has `venv`; set `CORTEX_PYTHON` to choose one); otherwise it downloads `uv` (checksum-verified) and installs a private Python 3.12 under `~/.cortex/install`, without admin rights or a package manager. Source installs also need Git, and Bun (which the installer installs when missing).

## Recommended Install

**macOS and Linux:**

```bash
# One-line installer (latest GitHub release)
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash

# Install release X.Y.Z, newer or older than the one you have (the vX.Y.Z GitHub release's wheel)
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
2. Uses a system Python 3.11+ (see above), or installs a private one with `uv`
3. Creates an isolated runtime under `~/.cortex/install`
4. Downloads the `cortex-llm` wheel from the GitHub release (the single distribution channel), verifies it against the release's `.sha256` asset, and installs the verified wheel — or installs the source checkout when run from a clone
5. Builds the OpenTUI sidecar when missing or stale (installing Bun if needed; release wheels already bundle it)
6. Links the `cortex` launcher into `~/.local/bin` and, when that folder is not on your `PATH`, adds it in your shell's profile (`~/.zshrc`; for bash both `~/.bashrc` and the login profile, `~/.bash_profile` or `~/.profile`; fish's `conf.d`; otherwise `~/.profile`); set `CORTEX_NO_MODIFY_PATH=1` to leave profiles alone

A checksum mismatch or a missing `.sha256` asset aborts the install — nothing is installed. If no release has been published yet, the installer says so and points at the source-checkout install instead. Prerelease versions are not installable via `install.sh` — pins are stable `X.Y.Z` only (`vX.Y.Z` works too).

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
bun install
bun run typecheck
bun run build      # sidecar binary for this machine's platform
```

## Verification

```bash
cortex
```

Then inside Cortex:

- `/status` — the active model and the model that answered the last turn
- `/login` — the provider picker, which shows which providers have a key configured

## Getting a Model

Add a key for a provider, then pick a model:

```text
/login openai <api_key>        # or anthropic / azure / openai-compatible
/model openai:gpt-5.5
```

Provider keys can also come from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`); see the [CLI Documentation](cli.md#login--credentials) for Azure and OpenAI-compatible endpoints. Cortex remembers the last selected model across restarts (`~/.cortex/state.yaml`).

## Upgrading

Cortex checks once a day (in the background, never blocking startup) whether a newer Cortex release exists and posts a one-line notice at session start when one does. Opt out with `auto_update_check: false` in `~/.cortex/config.yaml`.

To go back to an earlier release, run the installer with that version
(`... | bash -s -- X.Y.Z`, as above): settings, keys and conversations stay.
`/update cortex` only ever moves to a newer release.

```text
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

**OpenTUI sidecar missing (source checkout)** — run `./install.sh` at the repository root, or `bun install && bun run build` in `frontend/cortex-tui`.

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
# Cortex 1.x also linked the launcher into a system folder; remove that link if it is there
for link in /opt/homebrew/bin/cortex /usr/local/bin/cortex; do
  if [ "$(readlink "$link")" = "$HOME/.local/bin/cortex" ]; then rm -f "$link"; fi
done
rmdir ~/.local/bin ~/.local 2>/dev/null  # only if nothing else is in them
```

That removes the program; your settings, saved keys, logs and conversations
stay in `~/.cortex`. To remove them too:

```bash
rm -rf ~/.cortex
```

Keys `/login` saved to `~/.cortex/credentials.json` go with `~/.cortex`. If
it said it saved a key to the system keyring instead, that key stays there
(service `cortex-llm`); delete it with your keyring tool: Keychain Access on
macOS, or on Linux Passwords and Keys (Seahorse) or, where it is installed,
`secret-tool clear service cortex-llm username openai_api_key` (likewise
`anthropic_api_key`, `azure_openai_api_key`, `openai_compatible_api_key`).

The installer may have put `~/.local/bin` on your `PATH`: a line under
`# Added by the Cortex installer` in `~/.zshrc`, `~/.bashrc`, `~/.bash_profile`
or `~/.profile`, or the file `~/.config/fish/conf.d/cortex.fish`. These
commands remove what it added, keep your own lines, and delete a profile file
the installer created once nothing else is in it (skip them if other programs
you use live in `~/.local/bin`):

These use `perl`, which macOS and most Linux systems have.

```bash
for f in ~/.zshrc ~/.bashrc ~/.bash_profile ~/.profile; do
  [ -f "$f" ] || continue
  perl -0pi -e 's/\n?# Added by the Cortex installer\n[^\n]*\n?//g' "$f"
  grep -q '[^[:space:]]' "$f" || rm -f "$f"
done
rm -f ~/.config/fish/conf.d/cortex.fish
```

`/update cortex` in Cortex 2.0.2 and earlier left a pip cache in `~/.cache/pip`;
`rm -rf ~/.cache/pip` removes it (it is shared with other Python tools).

On Windows, also delete `%LOCALAPPDATA%\Programs\Cortex` and remove it from
your user `PATH`.

If you installed into your own environment instead: `pip uninstall cortex-llm`.

## Next Steps

1. Start Cortex: `cortex`
2. Get a model: `/login` + `/model provider:model`
3. Ask for changes to your code — see the [CLI Documentation](cli.md)
4. Tune settings — see the [Configuration Guide](configuration.md)
