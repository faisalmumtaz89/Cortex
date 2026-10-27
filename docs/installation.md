# Installation Guide

## Supported Systems

| System | Notes |
|---|---|
| macOS 13 or later, Apple Silicon or Intel | |
| Linux x86_64 or arm64 with glibc 2.28 or later | Debian 10, Ubuntu 20.04, RHEL 8, Fedora, Arch, or newer. Not musl distributions such as Alpine. |
| Windows 10 or 11 | Through WSL (Windows Subsystem for Linux) with a Linux distribution such as Ubuntu; the PowerShell installer installs Cortex into it. |

The installer needs only `curl`. It uses a Python 3.11+ from your system when one is there (the first of `python3.13`, `python3.12`, `python3.11` and `python3` that has `venv`; set `CORTEX_PYTHON` to choose one); otherwise it downloads `uv` (checksum-verified) and installs a private Python 3.12 under `~/.cortex/install`, without admin rights or a package manager. The private Python and `uv` stay in `~/.cortex/install`: neither adds anything to `~/.local/bin` or `~/.cache`.

Running the installer again (to upgrade, say) keeps the Python the runtime already runs on, while it still works. To change it, choose for that run: `CORTEX_MANAGED_PYTHON=1` for the private Python, `CORTEX_MANAGED_PYTHON=0` or `CORTEX_PYTHON=/path/to/python3` for one from your system (see [Installer Variables](#installer-variables)). The installer then builds the runtime again on that Python, and removes the private Python and `uv` once nothing runs on them, saying so. Its `→ Using Python …` line names the Python it uses and, when it kept the runtime's, says so. Source installs also need Git, and Bun (which the installer installs when missing).

## Recommended Install

**macOS and Linux:**

```bash
# One-line installer (latest GitHub release)
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash

# Install release X.Y.Z, newer or older than the one you have (the vX.Y.Z GitHub release's wheel)
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash -s -- X.Y.Z

# Install on the private Python: an installer variable goes after the pipe, before bash
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | CORTEX_MANAGED_PYTHON=1 bash

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
3. Creates an isolated runtime under `~/.cortex/install`, and makes `~/.cortex` readable only by you (it will hold your keys and conversations)
4. Downloads the `cortex-llm` wheel from the GitHub release (the single distribution channel), verifies it against the release's `.sha256` asset, and installs the verified wheel — or installs the source checkout when run from a clone
5. Builds the OpenTUI sidecar when missing or stale (installing Bun if needed; release wheels already bundle it)
6. Links the `cortex` launcher into `~/.local/bin` and, when that folder is not on your `PATH`, adds it in your shell's profile (`~/.zshrc`; for bash both `~/.bashrc` and the login profile, `~/.bash_profile` or `~/.profile`; fish's `conf.d`; otherwise `~/.profile`); set `CORTEX_NO_MODIFY_PATH=1` to leave profiles alone

A checksum mismatch or a missing `.sha256` asset aborts the install — nothing is installed. If no release has been published yet, the installer says so and points at the source-checkout install instead. Prerelease versions are not installable via `install.sh` — pins are stable `X.Y.Z` only (`vX.Y.Z` works too).

### Installer Variables

Each applies to the run it is set for. With `curl ... | bash`, set it on `bash`, the side of the pipe that runs the installer (`curl ... | CORTEX_MANAGED_PYTHON=1 bash -s -- X.Y.Z`): set before `curl`, it reaches only `curl`, and the installer runs without it.

| Variable | Effect |
|---|---|
| `CORTEX_MANAGED_PYTHON=1` | Use the private Python, installing it when missing. |
| `CORTEX_MANAGED_PYTHON=0` | Use a Python from your system; stop if there is none. |
| `CORTEX_PYTHON=/path/to/python3` | Use this Python; stop, with the reason, if it is not 3.11 or later with `venv`. |
| `CORTEX_NO_MODIFY_PATH=1` | Change no shell profile; the installer prints the full path to run Cortex by. |

A choice that cannot be met, or a value other than `1` or `0`, stops the installer before it changes anything.
| `$env:CORTEX_VERSION = "X.Y.Z"` | Windows (`install.ps1`): install that release instead of the latest. |

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

- `/status` — the Cortex version, the directory, the active model and the model that answered the last turn
- `/login` — the provider picker, which shows which providers have a key configured

## Getting a Model

Add a key for a provider, then pick a model:

```text
/login openai <api_key>                          # or anthropic
/login openai-compatible <api_key> <base_url>    # azure: <api_key> <endpoint>
/model openai:gpt-5.5
```

Typing `/login` alone opens a picker of the providers. Provider keys can also come from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY` with `AZURE_OPENAI_ENDPOINT`, `OPENAI_COMPATIBLE_API_KEY` with `OPENAI_COMPATIBLE_BASE_URL`), and on a machine that only runs `cortex -p`, `cortex login PROVIDER [URL]` saves one (see [Headless Mode](cli.md#saving-a-key-without-the-interactive-app)). Cortex remembers the last selected model across restarts (`~/.cortex/state.yaml`).

## Upgrading

Cortex checks once a day (in the background, never blocking startup) whether a newer Cortex release exists and posts a one-line notice at session start when one does. Opt out with `auto_update_check: false` in `~/.cortex/config.yaml`.

```text
/update            # inside Cortex: the installed version, and a newer one if there is one
/update cortex     # install the latest published Cortex release (restart to apply)
```

Or re-run the installer from a shell. It keeps the Python Cortex runs on,
and your settings, keys and conversations:

```bash
# The latest release
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash

# Release X.Y.Z, newer or older than the one you have
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash -s -- X.Y.Z
```

`/update cortex` only ever moves to a newer release: to go back to an earlier
one, use the installer.

`/update cortex` only ever installs a pinned, published release that is strictly newer than the running version: it downloads that GitHub release's wheel asset, verifies it against the release's `.sha256` asset, and installs the verified wheel into the running environment. The new version applies when you restart Cortex. Quitting Cortex while the install step is running is safe — the worker finishes that step before exiting rather than interrupting it.

Running from a source checkout (`git clone` + `./install.sh`)? `/update cortex` refuses and points you at `git pull` instead — installing the release wheel would replace your editable install.

## Common Installation Issues

**OpenTUI sidecar missing (source checkout)** — run `./install.sh` at the repository root, or `bun install && bun run build` in `frontend/cortex-tui`.

**Permission denied on `~/.cortex`**

```bash
sudo chown -R $(whoami) ~/.cortex
chmod -R u+rwX ~/.cortex
```

For everything else, see the [Troubleshooting Guide](troubleshooting.md).

## Uninstallation

Uninstalling has up to three parts: the program, your data in `~/.cortex`,
and the `PATH` line in your shell profiles. Each can be done alone.

### The Program

```bash
# Remove the isolated runtime and launcher
rm -rf ~/.cortex/install
rm -f ~/.local/bin/cortex
# Cortex 1.x also linked the launcher into a system folder; remove that link if it is there
for link in /opt/homebrew/bin/cortex /usr/local/bin/cortex; do
  if [ "$(readlink "$link")" = "$HOME/.local/bin/cortex" ]; then rm -f "$link"; fi
done
rmdir ~/.local/bin ~/.local 2>/dev/null || true  # only if nothing else is in them
```

That removes the program; your settings, saved keys, logs and conversations
stay in `~/.cortex`.

### Your Data

To remove your data but keep a `~/.cortex/config.yaml` you wrote (and the
program, if it is still in `~/.cortex/install`), remove everything else:
saved keys, logs, conversations, the remembered model, and the "Allow
always" answers in `tool_permissions.yaml`:

```bash
find ~/.cortex -mindepth 1 -maxdepth 1 ! -name config.yaml ! -name install -exec rm -rf {} +
```

Run it after removing the program, too: a Cortex run in between writes a
log and state again. A `~/.cortex` you keep stays readable only by you.

To remove all of `~/.cortex` instead, `config.yaml` and the program
included: `rm -rf ~/.cortex`.

Keys `/login` saved to `~/.cortex/credentials.json` go with `~/.cortex`. If
it said it saved a key to the system keyring instead, that key stays there
(service `cortex-llm`); delete it with your keyring tool: Keychain Access on
macOS, or on Linux Passwords and Keys (Seahorse) or, where it is installed,
`secret-tool clear service cortex-llm username openai_api_key` (likewise
`anthropic_api_key`, `azure_openai_api_key`, `openai_compatible_api_key`).

### The PATH Line

The installer may have put `~/.local/bin` on your `PATH`: a line under
`# Added by the Cortex installer` in `~/.zshrc`, `~/.bashrc`, `~/.bash_profile`
or `~/.profile`, or the file `~/.config/fish/conf.d/cortex.fish`. The commands
below remove only what it added: your own lines stay, and a profile file goes
only if nothing else is left in it. Skip them if you run other programs from
`~/.local/bin` by name, including ones that only became reachable when the
installer added it: after the cleanup, new terminals do not find them.
They use `perl`, which macOS and most Linux systems have.

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
