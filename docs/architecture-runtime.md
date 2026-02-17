# Runtime Architecture (OpenTUI + Python Worker)

## Status
- Accepted
- Date: 2026-02-13

## Decision
Cortex now uses a hybrid runtime:

1. Frontend terminal rendering is owned by an OpenTUI sidecar.
2. Backend inference/tooling remains in Python.
3. Frontend and backend communicate via line-delimited JSON-RPC 2.0 plus structured event frames.

## Why
Legacy rendering mixed Rich `Live` updates and raw ANSI/stdout writes from multiple paths, which created duplicate or unstable output under streaming/tooling flows. The split enforces a single terminal writer and separates UI concerns from inference/tool execution.

## Components
- Frontend package: `/Users/faisalmumtaz/Documents/GitHub/Cortex/frontend/cortex-tui`
- Worker mode entry: `python -m cortex --worker-stdio`
- Protocol server: `/Users/faisalmumtaz/Documents/GitHub/Cortex/cortex/protocol/rpc_server.py`
- Event envelope types: `/Users/faisalmumtaz/Documents/GitHub/Cortex/cortex/protocol/types.py`
- Worker runtime assembly: `/Users/faisalmumtaz/Documents/GitHub/Cortex/cortex/app/worker_runtime.py`
- Launcher: `/Users/faisalmumtaz/Documents/GitHub/Cortex/cortex/ui_runtime/launcher.py`

## Protocol Contract
Each event frame includes:
- `session_id`
- `seq` (monotonic per session)
- `ts_ms`
- `event_type`
- `payload`

Supported event types:
- `session.status`
- `message.updated`
- `message.part.updated`
- `permission.asked`
- `permission.replied`
- `session.error`
- `system.notice`

Protocol version is strict (`1.0.0`). Handshake mismatch fails fast.

## Worker Output Safety
Worker mode reserves real stdout for JSON-RPC frames and redirects normal `print`/stdout writes away from the RPC channel. This prevents accidental protocol corruption from backend logging or legacy print paths.

## Current Scope
- Supported target for migration: `darwin-arm64`.
- Local inference/fine-tuning remain Python-first (MLX/PyTorch/GGUF).
- OpenTUI is default launch target via `cortex`.
- Legacy Rich UI remains available as explicit compatibility mode (`--legacy-ui`) while command parity is completed.

## Non-Goals (Current Phase)
- MCP/plugin loading
- Multi-agent/subtask runtime
- Full shell/write tooling policy stack beyond current staged tooling controls
