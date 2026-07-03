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
