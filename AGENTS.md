# Cortex — Agent & Contributor Guide

Cortex is a terminal AI coding agent for Apple Silicon: local models via MLX
(primary) and GGUF (llama.cpp), plus OpenAI/Anthropic cloud models, driving a
small permissioned tool set (read_file, list_dir, search, edit_file,
write_file, bash) inside the user's repository.

## Architecture (two processes)

- `frontend/cortex-tui/` — OpenTUI (Bun + SolidJS) terminal frontend. It spawns
  the worker and renders its event stream.
- `python -m cortex --worker-stdio` — Python worker: JSON-RPC 2.0, one frame
  per line on stdio. Entry: `cortex/__main__.py` → `cortex/app/worker_runtime.py`.
- `python -m cortex -p "..."` — headless single turn through the same worker
  wiring (`cortex/app/headless.py`).

Key packages: `cortex/app/` (services), `cortex/tooling/` (agent loop, tools,
permissions, system prompt), `cortex/cloud/` (providers + router),
`cortex/protocol/` (RPC schema/events), `cortex/metal/` + `inference_engine.py`
+ `model_manager.py` (local inference).

## The prime rule: runtime evidence over reasoning

Behavioral claims must be validated against the real runtime, not inferred
from reading code. The empirical gates, in order of authority:

1. `tests/test_agent_runtime_e2e.py` — spawns the real worker subprocess in a
   scratch repo and drives full agent turns over JSON-RPC with a deterministic
   scripted model (`CORTEX_SCRIPTED_MODEL`). Asserts observable effects:
   files on disk, event sequences, exit codes.
2. `tests/test_tui_e2e.py` — runs the REAL TUI (sidecar binary + worker) in
   tmux with the scripted model and asserts on captured frames: rendering
   order, folds, permission modal contents, real file side effects, spinner,
   input queueing. Any TUI behavior change needs a frame assertion here;
   rebuild the sidecar first (`cd frontend/cortex-tui && bun run build`).
3. The full suite: `.venv/bin/python -m pytest tests/ -q` — must be green
   before and after any change.
4. For manual verification: `python -m cortex -p "prompt" --model ...`
   exercises a real turn end to end and prints it.

If a change cannot be observed through one of these, add the scenario to the
E2E suite first, then make the change.

## Conventions

- Python ≥ 3.11, line length 100, `ruff` + `mypy` clean (`./scripts/typecheck.sh`).
- Never print to stdout in worker code — stdout is the JSON-RPC channel.
  Diagnostics go to stderr or the log file.
- Tools must stay sandboxed to the repo root (`resolve_repo_path`) and every
  mutating permission (`edit`, `bash`) must flow through `PermissionManager`.
- No new dependencies without a strong reason; leanness is a feature.
- Delete dead code instead of commenting it out.
