# Benchmarking Cortex on Terminal-Bench 2.0

The empirical scoreboard for Cortex-as-a-coding-agent: real tasks, real
containers, observable pass/fail — the same doctrine as the E2E suite, scaled
up.

## Setup (once)

```bash
# Harness (needs Docker running)
python3 -m venv ~/.harbor-venv && ~/.harbor-venv/bin/pip install harbor

# Lean cloud-only wheel for the task containers
./benchmark/build_wheel.sh
```

## Run

```bash
export AZURE_OPENAI_API_KEY=...   # or OPENAI_API_KEY / ANTHROPIC_API_KEY
export AZURE_OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com

# One task
PYTHONPATH=benchmark ~/.harbor-venv/bin/harbor run \
  -d terminal-bench@2.0 -a "cortex_agent:CortexAgent" -m azure/gpt-5.5 \
  -i chess-best-move -y

# Full 89-task set, 4 concurrent
PYTHONPATH=benchmark ~/.harbor-venv/bin/harbor run \
  -d terminal-bench@2.0 -a "cortex_agent:CortexAgent" -m azure/gpt-5.5 -n 4 -y
```

Results land in `jobs/<timestamp>/`: job-level `result.json`, per-trial
`result.json`, agent transcript at `<trial>/agent/cortex.txt`, verifier output
under `<trial>/verifier/`.

## How the adapter works

`benchmark/cortex_agent.py` implements Harbor's `BaseInstalledAgent`:

- `install()` bootstraps python3 (>=3.11) in the task container, uploads the
  wheel from `benchmark/dist/`, and installs it cloud-only (`--no-deps` +
  minimal deps) into `/installed-agent/venv`.
- `run()` executes one headless turn:
  `cortex -p "<instruction>" --model azure:gpt-5.5 --full-auto`, with runtime
  tuning via `CORTEX_*` env-config overrides (iterations, timeouts, max
  tokens).

Rebuild the wheel (`./benchmark/build_wheel.sh`) after changing Cortex source.
