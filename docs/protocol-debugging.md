# Protocol Debugging Guide

## Worker Transport

- Worker mode: `python -m cortex --worker-stdio`
- Transport: line-delimited JSON-RPC 2.0 frames over stdio
- Events are emitted as JSON-RPC notifications (`method: "event"`).

## Handshake

Send:

```json
{"jsonrpc":"2.0","id":1,"method":"app.handshake","params":{"protocol_version":"1.0.0"}}
```

Expected:

- Success with `protocol_version: "1.0.0"`
- Mismatch returns error `"Protocol version mismatch"`.

## Event Envelope

Every event includes:

- `session_id`
- `seq` (monotonic per session)
- `ts_ms`
- `event_type`
- `payload`

The frontend should drop stale/replayed events with `seq <= last_seq`.

## Failure Taxonomy

- Parse errors: JSON-RPC `-32700`
- Invalid request/params: JSON-RPC `-32600` / `-32602`
- Method not found: JSON-RPC `-32601`
- Internal worker failure: JSON-RPC `-32603`
- Protocol mismatch: JSON-RPC `-32000`
- Permission reply missing/stale: JSON-RPC `-32001`

## Logs

Key log dimensions in `~/.cortex/cortex.log`:

- `request_id`, `provider`, `model`, `attempt` (cloud routing)
- `session_id`, `seq`, `event_type` (worker event emission)

## Smoke Checks

1. Handshake only:

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"app.handshake","params":{"protocol_version":"1.0.0"}}' \
| python -m cortex --worker-stdio
```

2. Verify stdout purity:

- stdout should contain only JSON frames
- non-protocol diagnostics must go to stderr

3. Full turn without a real model: set `CORTEX_SCRIPTED_MODEL` to a JSON script (`{"responses": [...]}`) and drive the worker over stdio — this is exactly what `tests/test_agent_runtime_e2e.py` does, and its `WorkerHarness` is the reference client implementation.

4. One-shot end-to-end check without the TUI: `python -m cortex -p "prompt"` (tool activity on stderr, reply on stdout).
