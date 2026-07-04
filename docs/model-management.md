# Model Management

Cortex delegates ALL local inference to **Lumen** (github.com/faisalmumtaz89/Lumen) —
a single Rust binary that runs Qwen3.5/3.6-family models GPU-resident on Apple
Silicon and serves an OpenAI-compatible API with native tool calls. Cortex owns
the full lifecycle; you never run Lumen by hand.

## Supported local models

Exactly what Lumen supports (`lumen models`): the Qwen3.5/3.6 family
(`qwen3-5-9b`, `qwen3-6-27b`, `qwen3-5-moe-35b-a3b`) in BF16 / Q8_0 / Q4_0.
Selectors are `name:quant`, e.g. `qwen3-5-9b:q4_0`.

## Lifecycle (all automatic)

- `/model` opens the picker: cached models are loadable now; others show
  `download required`.
- Selecting a local model starts a managed `lumen-server` on a free localhost
  port (first token ~15–20s later, GPU-resident after that). Switching models
  restarts the server; quitting Cortex terminates it.
- `/download <model[:quant]>` runs `lumen pull` with streamed progress.
- `/status` shows the running server (model, port); logs at
  `~/.cortex/lumen-server.log`.
- If the Lumen binaries are missing, Cortex says so:
  `curl -fsSL https://servelumen.com/install.sh | bash`.

Cloud models (OpenAI / Anthropic / Azure) are unchanged — see `docs/cli.md`.
