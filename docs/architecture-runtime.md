# Runtime Architecture (OpenTUI + Python Worker)

Cortex is split into two processes with a strict protocol boundary:

1. **Frontend** — OpenTUI sidecar (Bun + SolidJS, `frontend/cortex-tui`) owns all terminal rendering. It spawns the worker and reconciles its event stream.
2. **Backend** — Python worker (`python -m cortex --worker-stdio`) owns models, sessions, and tools.

They communicate over line-delimited JSON-RPC 2.0 on the worker's stdio, plus structured event frames emitted as JSON-RPC notifications. The split enforces a single terminal writer and separates UI concerns from inference/tool execution.

## Components

- Frontend package: `frontend/cortex-tui`
- Launcher: `cortex/ui_runtime/launcher.py` (prefers the live TS entrypoint in a source checkout, otherwise the bundled `cortex/ui_runtime/bin/cortex-tui` binary)
- Worker entry: `cortex/__main__.py` → `cortex/app/worker_runtime.py`
- Protocol server and contracts: `cortex/protocol/rpc_server.py`, `cortex/protocol/schema.py`, `cortex/protocol/types.py`
- Services: `cortex/app/session_service.py`, `cortex/app/command_service.py`, `cortex/app/model_service.py`, `cortex/app/permission_service.py`
- Agent loop: `cortex/tooling/orchestrator.py` with `cortex/tooling/registry.py` (tools), `cortex/tooling/permissions.py` (rules), `cortex/tooling/agent_prompt.py` (system prompt + `AGENTS.md` project context)
- Local inference: `cortex/inference_engine.py`, `cortex/model_manager.py`, `cortex/metal/` (MLX + GGUF)
- Cloud inference: `cortex/cloud/` (OpenAI/Anthropic clients + router)

## Headless Mode

`python -m cortex -p "prompt" [--model <selector>] [--full-auto]` (`cortex/app/headless.py`) runs one agent turn through the same `WorkerRuntime` wiring the TUI uses. Assistant text streams to stdout; tool activity and errors go to stderr. Permission policy is rule-based instead of interactive: reads allowed, mutations denied unless `--full-auto`. Exit codes: `0` success, `1` turn error, `2` setup error.

## Agent Turn Flow

1. The frontend submits user input via `session.submit_user_input`.
2. `SessionService` builds the turn and hands it to the `ToolingOrchestrator`, which assembles the system prompt (identity, working directory, `AGENTS.md`/`CLAUDE.md` project context, and — for local models — the `<tool_calls>` protocol from `cortex/tooling/local_protocol.py`).
3. The model streams text and tool calls. Cloud models use native tool calling; local models emit a `<tool_calls>` JSON block that the orchestrator parses.
4. Each tool call passes through the `PermissionManager`. Reads are allowed by default rules; `edit`/`bash` permissions trigger a `permission.asked` event, which the frontend answers via the `permission.reply` method (allow once / allow always / reject).
5. Tool results are fed back to the model until it answers without tool calls, up to `tools_max_iterations`.

## Protocol Contract

Each event frame includes:

- `session_id`
- `seq` (monotonic per session)
- `ts_ms`
- `event_type`
- `payload`

Supported event types: `session.status`, `message.updated`, `message.part.updated`, `permission.asked`, `permission.replied`, `session.error`, `system.notice`.

Protocol version is strict (`1.0.0`). Handshake mismatch fails fast.

## Worker Output Safety

Worker mode reserves real stdout for JSON-RPC frames and redirects normal `print`/stdout writes to stderr. This prevents accidental protocol corruption from backend logging or stray print paths.

## Empirical Validation

`tests/test_agent_runtime_e2e.py` spawns the real worker subprocess in a scratch repository and drives full agent turns over JSON-RPC. Only the model is replaced: setting `CORTEX_SCRIPTED_MODEL` to a JSON script path activates a deterministic scripted client (`cortex/cloud/clients/scripted_client.py`), so the orchestrator, tool registry, permission engine, event stream, and persistence all run for real. Assertions check observable effects — files on disk, event sequences, exit codes.

## Scope

- Target platform: `darwin-arm64` only.
- Local inference: MLX (primary) and GGUF (llama.cpp).
- OpenTUI is the only interactive runtime; headless mode is the only non-interactive one.

## Non-Goals (Current Phase)

- MCP/plugin loading
- Multi-agent/subtask runtime
