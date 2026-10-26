# Development Guide

## Getting Started

```bash
git clone https://github.com/faisalmumtaz89/Cortex.git
cd Cortex

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### Prerequisites

- macOS (Apple Silicon or Intel) or Linux (x86_64 or arm64, glibc)
- Python 3.11 or later
- Bun runtime for frontend work (global `bun`, or the local runtime provisioned by `npm install` in `frontend/cortex-tui`)

`bun run build` compiles the sidecar for the machine it runs on (OpenTUI's native core comes from the platform package `bun install` fetches); set `CORTEX_TUI_TARGET` to a Bun target to override. The TUI E2E tests need `tmux`.

## Project Structure

```
Cortex/
├── pyproject.toml                  # Build config, dependencies, tool settings
├── config.yaml                     # Commented configuration template
├── setup.py                        # Tags each wheel for the platform its sidecar was built for
├── install.sh                      # macOS/Linux installer (isolated runtime, private Python if needed)
├── install.ps1                     # Windows installer (installs inside WSL, adds a cortex command)
├── cortex/
│   ├── __main__.py                 # Entry point: TUI launch, --worker-stdio, -p headless
│   ├── config.py                   # Configuration loading (Config class)
│   ├── conversation_manager.py     # Chat history and persistence
│   ├── logging_config.py           # Log file setup
│   ├── runtime_io.py               # Worker stdio hygiene
│   ├── update_check.py             # Release discovery for /update
│   ├── app/                        # Worker application service layer
│   │   ├── worker_runtime.py       # JSON-RPC runtime assembly
│   │   ├── session_service.py      # Session orchestration
│   │   ├── command_service.py      # Slash command execution
│   │   ├── command_output.py       # Slash command output formatting
│   │   ├── model_service.py        # Model/auth operations
│   │   ├── permission_service.py   # Permission ask/reply bridge
│   │   ├── update_service.py       # /update status and self-update
│   │   └── headless.py             # cortex -p single-turn execution
│   ├── tooling/                    # Agent loop
│   │   ├── orchestrator.py         # Generation + tool execution + permissions
│   │   ├── registry.py             # Profile-aware tool registry
│   │   ├── permissions.py          # Rule engine + persisted approvals
│   │   ├── provenance.py           # Per-turn model provenance verification
│   │   ├── agent_prompt.py         # System prompt + AGENTS.md context + skill list
│   │   ├── skills.py               # Skill discovery and SKILL.md parsing
│   │   └── builtin/                # read_file, list_dir, search, edit_file, write_file, bash, skill
│   ├── cloud/                      # Model providers
│   │   ├── router.py               # Provider routing
│   │   ├── retry.py                # Request retries and backoff
│   │   ├── catalog.py              # Model catalog (+ ~/.cortex/cloud_models.json overrides)
│   │   ├── credentials.py          # Env, keyring, and key-file credential handling
│   │   └── clients/                # openai, anthropic, chat_completions, scripted clients
│   ├── skills/                     # Built-in skills (one folder with SKILL.md each)
│   ├── protocol/                   # JSON-RPC contracts/server (rpc_server, schema, types, events)
│   └── ui_runtime/                 # OpenTUI launcher, bundled sidecar binary, platforms.py (the shipped platforms)
├── frontend/cortex-tui/            # OpenTUI + Solid frontend source
├── tests/                          # Pytest suite (incl. test_agent_runtime_e2e.py)
├── scripts/typecheck.sh            # mypy wrapper
├── scripts/smoke_install.sh        # Installs a built wheel like a new user and runs the CLI, a turn and the TUI
└── docs/
```

## Runtime Split

- Frontend: OpenTUI sidecar (`frontend/cortex-tui`), spawned by `cortex`
- Backend: Python worker (`python -P -m cortex --worker-stdio`), JSON-RPC 2.0 over stdio
- Headless: `python -P -m cortex -p "..."` reuses the worker wiring for one turn

Never print to stdout in worker code — stdout is the JSON-RPC channel. Diagnostics go to stderr or the log file.

Frontend source commands:

```bash
cd frontend/cortex-tui
npm install
npm run typecheck
npm run dev        # Runs OpenTUI from source
npm run build      # Builds the sidecar binary for this machine's platform
```

## Testing: the Empirical Gate

Behavioral claims must be validated against the real runtime, not inferred from reading code:

1. **E2E suite** — `tests/test_agent_runtime_e2e.py` spawns the real worker subprocess in a scratch repository and drives full agent turns over JSON-RPC. The model is replaced with a deterministic script (`CORTEX_SCRIPTED_MODEL` pointing at a JSON script file); everything else — orchestrator, tool registry, permission engine, event stream, persistence — runs for real. Assertions check observable effects: files on disk, event sequences, exit codes.
2. **TUI E2E** — `tests/test_tui_e2e.py` runs the real TUI (sidecar + worker) in tmux with the scripted model and asserts on captured frames. It is skipped without tmux or a built sidecar; rebuild first with `cd frontend/cortex-tui && bun run build`.
3. **Full suite** — `python -m pytest tests/ -q` must be green before and after any change.
4. **Manual verification** — `python -P -m cortex -p "prompt" --model ...` exercises a real turn end to end.

If a change cannot be observed through one of these, add the scenario to the E2E suite first, then make the change.

```bash
python -m pytest tests/ -q                          # everything
python -m pytest tests/test_agent_runtime_e2e.py -q # just the worker E2E gate
python -m pytest tests/test_tui_e2e.py -q           # just the TUI E2E gate
```

## Code Style

Configured in `pyproject.toml`: line length 100, `ruff` (rules E, F, I, N, W) and `mypy` must be clean.

```bash
ruff check cortex/ tests/
./scripts/typecheck.sh        # mypy cortex
```

## Key Classes Reference

| Class | Module | Purpose |
|---|---|---|
| `Config` | `cortex.config` | Loads and holds runtime configuration |
| `ConversationManager` | `cortex.conversation_manager` | Chat history and persistence |
| `WorkerRuntime` | `cortex.app.worker_runtime` | JSON-RPC session runtime |
| `CommandService` | `cortex.app.command_service` | Slash command execution |
| `ToolingOrchestrator` | `cortex.tooling.orchestrator` | Agent loop: generation, tools, permissions |
| `ToolRegistry` | `cortex.tooling.registry` | Profile-aware tool registry |
| `PermissionManager` | `cortex.tooling.permissions` | Permission rules and persisted approvals |
| `CloudRouter` | `cortex.cloud.router` | Routes turns to the selected provider's client |

## Releasing

Pushing a tag `vX.Y.Z` runs `.github/workflows/release.yml` (one release at a time via a `concurrency` group; re-runs are idempotent — the action updates the existing release by tag and replaces same-named assets). It builds one wheel per platform in `cortex/ui_runtime/platforms.py`, each on a runner of that platform (`macos-14`, `macos-15-intel`, `ubuntu-24.04`, `ubuntu-24.04-arm`):

1. Verifies the tag matches `pyproject.toml`'s `version`.
2. Builds the OpenTUI sidecar for the runner's platform with a **pinned Bun version** (kept in sync with `frontend/cortex-tui/package.json`), then the wheel with `python -m build --wheel`; `setup.py` tags it for that platform.
3. **Hard gates:** the job fails unless the wheel contains `cortex/ui_runtime/bin/cortex-tui` (the sidecar is gitignored, so a wheel built without the bun step would silently ship a TUI-less package) AND carries that platform's tag (the wheel bundles a native binary and must never be `any`).
4. Generates the wheel's `.sha256` sibling (`<hex>  <filename>`) — installers refuse a wheel without it.
5. Runs `scripts/smoke_install.sh`: `install.sh` installs the wheel from a local copy laid out like the release, then `cortex --version`, a headless agent turn with the scripted model, and the TUI's first screen must work. On one macOS and one Linux runner it runs again with `install.sh`'s private Python.
6. A Windows job lints `install.ps1` and checks that it explains the WSL setup on a machine without a Linux distribution.
7. **Last**, once every platform passed, creates the GitHub release with the four wheels and their `.sha256` files attached.

Running the workflow by hand (Actions → Release → Run workflow) is a dry run: it builds and smoke-tests every platform and publishes nothing. `.github/workflows/ci.yml` runs the test suite on all four platforms on every push.

**Single-channel design:** GitHub Releases is the only distribution channel. `install.sh` and `/update cortex` *discover* the latest version through GitHub's `releases/latest` redirect and *install* the wheel asset from that same release, verifying it against the `.sha256` sibling before `pip install`. Because discovery and install source are one artifact, a version is either fully live (release exists, assets attached, gates passed) or does not exist — the dual-channel failure class where a release is discoverable on one channel but not yet (or never) installable on another cannot occur. This also makes the old package-index history irrelevant: the tag gate only has to match `pyproject.toml`, it never has to out-version a stale index entry.

Notes:

- The `.sha256` verification protects **integrity** (truncation, corruption, wrong asset), not authenticity beyond HTTPS — the checksum shares the wheel's origin, which is why installers hard-refuse any non-`https://github.com` asset base.
- `pip install <wheel>` still resolves *new or changed dependencies* from the package index — single-channel applies to the `cortex-llm` artifact itself.
- Prerelease tags are not installable: PEP 440 normalizes suffixes out of wheel filenames (`1.2.3-rc1` → `1.2.3rc1`), so version→asset-URL reconstruction only supports stable `X.Y.Z`, and the update check never acts on prerelease tags.
- **Quitting during a self-update is safe, with one bounded caveat:** the `pip install <wheel>` step rewrites `cortex-llm` inside the very venv Cortex launches from, and killing pip mid-flight would skip its rollback and strand a half-removed install that cannot relaunch. So worker shutdown never signals an in-flight self-install pip — it waits (up to ~60s) for it to finish before exiting. Only a pip wedged past that bound is killed as a last resort; recovery from that worst case is re-running `install.sh` from a terminal. Wheel *downloads* interrupted by quitting are simply aborted — nothing is installed.
- **Source checkouts refuse `/update cortex`:** installing the release wheel would replace the editable `pip install -e` dist (and, through the `site-packages/cortex` symlink that `install.sh`'s source mode creates, could overwrite working-tree files). The plan step detects the checkout and answers with a `git pull` pointer instead; the startup update notice does the same. `CORTEX_SELF_INSTALL_KIND=installed` (test-only) forces the normal wheel path — the test suite itself runs from a checkout.

Bump `version` in `pyproject.toml` and `cortex/__init__.py.__version__` together before tagging — `/update cortex` compares the released tag against the installed version and only ever installs the pinned release.
