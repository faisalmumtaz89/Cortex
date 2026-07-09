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

- macOS on Apple Silicon (ARM64)
- Python 3.11 or later
- Bun runtime for frontend work (global `bun`, or the local runtime provisioned by `npm install` in `frontend/cortex-tui`)

## Project Structure

```
Cortex/
├── pyproject.toml                  # Build config, dependencies, tool settings
├── config.yaml                     # Commented configuration template
├── install.sh                      # Shell installer (isolated runtime + sidecar build)
├── cortex/
│   ├── __main__.py                 # Entry point: TUI launch, --worker-stdio, -p headless
│   ├── config.py                   # Configuration loading (Config class)
│   ├── gpu_validator.py            # Apple Silicon / Metal / MLX validation
│   ├── model_manager.py            # Local model loading (MLX + GGUF)
│   ├── model_downloader.py         # HuggingFace downloads with resume
│   ├── inference_engine.py         # Local text generation
│   ├── conversation_manager.py     # Chat history and persistence
│   ├── app/                        # Worker application service layer
│   │   ├── worker_runtime.py       # JSON-RPC runtime assembly
│   │   ├── session_service.py      # Session orchestration
│   │   ├── command_service.py      # Slash command execution
│   │   ├── model_service.py        # Model/auth operations
│   │   ├── permission_service.py   # Permission ask/reply bridge
│   │   └── headless.py             # cortex -p single-turn execution
│   ├── tooling/                    # Agent loop
│   │   ├── orchestrator.py         # Generation + tool execution + permissions
│   │   ├── registry.py             # Profile-aware tool registry
│   │   ├── permissions.py          # Rule engine + persisted approvals
│   │   ├── agent_prompt.py         # System prompt + AGENTS.md context
│   │   ├── local_protocol.py       # <tool_calls> protocol for local models
│   │   └── builtin/                # read_file, list_dir, search, edit_file, write_file, bash
│   ├── cloud/                      # Cloud providers
│   │   ├── router.py               # Provider routing + retries
│   │   ├── catalog.py              # Model catalog (+ ~/.cortex/cloud_models.json overrides)
│   │   ├── credentials.py          # Keychain/env credential handling
│   │   └── clients/                # openai_client, anthropic_client, scripted_client
│   ├── protocol/                   # JSON-RPC contracts/server (rpc_server, schema, types, events)
│   ├── metal/                      # MLX acceleration layer (accelerator, converter, memory pool, profiler)
│   ├── template_registry/          # Chat template detection and profiles
│   └── ui_runtime/                 # OpenTUI launcher + bundled sidecar binary
├── frontend/cortex-tui/            # OpenTUI + Solid frontend source
├── tests/                          # Pytest suite (incl. test_agent_runtime_e2e.py)
├── scripts/typecheck.sh            # mypy wrapper
└── docs/
```

## Runtime Split

- Frontend: OpenTUI sidecar (`frontend/cortex-tui`), spawned by `cortex`
- Backend: Python worker (`python -m cortex --worker-stdio`), JSON-RPC 2.0 over stdio
- Headless: `python -m cortex -p "..."` reuses the worker wiring for one turn

Never print to stdout in worker code — stdout is the JSON-RPC channel. Diagnostics go to stderr or the log file.

Frontend source commands:

```bash
cd frontend/cortex-tui
npm install
npm run typecheck
npm run dev        # Runs OpenTUI from source
npm run build      # Builds the darwin-arm64 bundled sidecar binary
```

## Testing: the Empirical Gate

Behavioral claims must be validated against the real runtime, not inferred from reading code:

1. **E2E suite** — `tests/test_agent_runtime_e2e.py` spawns the real worker subprocess in a scratch repository and drives full agent turns over JSON-RPC. The model is replaced with a deterministic script (`CORTEX_SCRIPTED_MODEL` pointing at a JSON script file); everything else — orchestrator, tool registry, permission engine, event stream, persistence — runs for real. Assertions check observable effects: files on disk, event sequences, exit codes.
2. **Full suite** — `python -m pytest tests/ -q` must be green before and after any change.
3. **Manual verification** — `python -m cortex -p "prompt" --model ...` exercises a real turn end to end.

If a change cannot be observed through one of these, add the scenario to the E2E suite first, then make the change.

```bash
python -m pytest tests/ -q                          # everything
python -m pytest tests/test_agent_runtime_e2e.py -q # just the E2E gate
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
| `GPUValidator` | `cortex.gpu_validator` | Validates Apple Silicon GPU availability |
| `ModelManager` | `cortex.model_manager` | Loads and manages local MLX/GGUF models |
| `ModelDownloader` | `cortex.model_downloader` | HuggingFace downloads with resume |
| `InferenceEngine` | `cortex.inference_engine` | Local text generation |
| `ConversationManager` | `cortex.conversation_manager` | Chat history and persistence |
| `WorkerRuntime` | `cortex.app.worker_runtime` | JSON-RPC session runtime |
| `CommandService` | `cortex.app.command_service` | Slash command execution |
| `ToolingOrchestrator` | `cortex.tooling.orchestrator` | Agent loop: generation, tools, permissions |
| `ToolRegistry` | `cortex.tooling.registry` | Profile-aware tool registry |
| `PermissionManager` | `cortex.tooling.permissions` | Permission rules and persisted approvals |
| `TemplateRegistry` | `cortex.template_registry` | Chat template detection and management |

## Releasing

Pushing a tag `vX.Y.Z` runs `.github/workflows/release.yml` on `macos-14` (one release at a time via a `concurrency` group; re-runs are idempotent — the action updates the existing release by tag and replaces same-named assets):

1. Verifies the tag matches `pyproject.toml`'s `version`.
2. Builds the OpenTUI sidecar with a **pinned Bun version** (kept in sync with `frontend/cortex-tui/package.json`), then the wheel with `python -m build --wheel`.
3. **Hard gates:** the job fails unless the wheel contains `cortex/ui_runtime/bin/cortex-tui` (the sidecar is gitignored, so a wheel built without the bun step would silently ship a TUI-less package) AND carries the `macosx…arm64` platform tag (declared in `setup.cfg` — the wheel bundles a Mach-O binary and must never be `any`).
4. Generates the wheel's `.sha256` sibling (shasum's `<hex>  <filename>` format) — installers refuse a wheel without it.
5. **Last**, creates the GitHub release with the wheel + `.sha256` attached.

**Single-channel design:** GitHub Releases is the only distribution channel. `install.sh` and `/update cortex` *discover* the latest version through GitHub's `releases/latest` redirect and *install* the wheel asset from that same release, verifying it against the `.sha256` sibling before `pip install`. Because discovery and install source are one artifact, a version is either fully live (release exists, assets attached, gates passed) or does not exist — the dual-channel failure class where a release is discoverable on one channel but not yet (or never) installable on another cannot occur. This also makes the old package-index history irrelevant: the tag gate only has to match `pyproject.toml`, it never has to out-version a stale index entry.

Notes:

- The `.sha256` verification protects **integrity** (truncation, corruption, wrong asset), not authenticity beyond HTTPS — the checksum shares the wheel's origin, which is why installers hard-refuse any non-`https://github.com` asset base.
- `pip install <wheel>` still resolves *new or changed dependencies* from the package index — single-channel applies to the `cortex-llm` artifact itself.
- Prerelease tags are not installable: PEP 440 normalizes suffixes out of wheel filenames (`1.2.3-rc1` → `1.2.3rc1`), so version→asset-URL reconstruction only supports stable `X.Y.Z`, and the update check never acts on prerelease tags.
- **Quitting during a self-update is safe, with one bounded caveat:** the `pip install <wheel>` step rewrites `cortex-llm` inside the very venv Cortex launches from, and killing pip mid-flight would skip its rollback and strand a half-removed install that cannot relaunch. So worker shutdown never signals an in-flight self-install pip — it waits (up to ~60s) for it to finish before exiting. Only a pip wedged past that bound is killed as a last resort; recovery from that worst case is re-running `install.sh` from a terminal. Wheel *downloads* interrupted by quitting are simply aborted — nothing is installed. (The Lumen installer is different: a kill there leaves Cortex itself untouched and the next `/update lumen` verifies and reports the mismatch, so it is reaped immediately.)
- **Source checkouts refuse `/update cortex`:** installing the release wheel would replace the editable `pip install -e` dist (and, through the `site-packages/cortex` symlink that `install.sh`'s source mode creates, could overwrite working-tree files). The plan step detects the checkout and answers with a `git pull` pointer instead; the startup update notice does the same. `CORTEX_SELF_INSTALL_KIND=installed` (test-only) forces the normal wheel path — the test suite itself runs from a checkout.

Bump `version` in `pyproject.toml` and `cortex/__init__.py.__version__` together before tagging — `/update cortex` compares the released tag against the installed version and only ever installs the pinned release.
