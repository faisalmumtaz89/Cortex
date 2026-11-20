# Running Cortex under Harbor

`cortex_agent.py` is a [Harbor](https://github.com/harbor-framework/harbor)
adapter: it runs a benchmark task, such as one of Terminal-Bench's, with a
released Cortex inside the task's container.

## What it does

- **Install:** runs the release's own `install.sh`, taken from the release's
  tag, with the version given, as the task's user. The installer checks the
  wheel against the release's checksum and does not touch the user's shell
  files. The image must have `bash`, a glibc userland (Cortex has no musl
  build, so not Alpine) and a home directory the task's user can write to;
  `curl` is installed if it is missing. The container needs to reach
  github.com and PyPI.
- **Run:** one headless turn,
  `cortex -p "<the task>" --model <provider:model> --full-auto`, as the
  task's user. Its output, the answer and the tool lines, is kept in the
  trial's logs as `agent/cortex.txt`.
- **Settings:** Cortex's documented maximums, since a task is one long turn:
  `CORTEX_TOOLS_MAX_ITERATIONS=1000`, `CORTEX_CLOUD_TIMEOUT_SECONDS=600`,
  `CORTEX_CLOUD_MAX_RETRIES=10`. Every other setting is Cortex's default.
  Pass a setting yourself (below) to change any of them, these three
  included.
- **Exit code:** Cortex's own. When the turn fails, Harbor records the error
  for the trial; the task's tests still run and decide the score.
- **A task whose text starts with `/`** is refused, with an error on the
  trial: Cortex would read it as a command.

Cortex prints no token counts, so Harbor's results carry none: meter the
model's use at the provider or with a proxy.

## Run it

Docker must be running. Install Harbor (`uv tool install harbor`), then, from
the repository's root:

```bash
PYTHONPATH=benchmark harbor run \
  -d terminal-bench/terminal-bench-2-1 \
  -a cortex_agent:CortexAgent --ak version=2.0.15 \
  -m openai-compatible:deepseek/deepseek-v4.1-flash \
  --ae OPENAI_COMPATIBLE_BASE_URL=https://example.com/v1 \
  --ae OPENAI_COMPATIBLE_API_KEY="$YOUR_KEY" \
  --ae CORTEX_REASONING_EFFORT=high
```

- `--ak version=` is the release to install, and is required: a run names
  what it measured.
- `-m` is the model as Cortex names it, `provider:model`.
- Each `--ae NAME=value` is set for Cortex in the container: the provider's
  key and URL (see [the CLI reference](../docs/cli.md)), and any `CORTEX_*`
  setting (see [Configuration](../docs/configuration.md)). A setting must be
  one the installed release has: Cortex ignores a `CORTEX_*` variable it
  does not know, and one whose value it does not accept falls back to the
  default (the run's log says so). `reasoning_effort` came with 2.0.15.

Results are written under `jobs/`: each trial has its `result.json`, the
agent's output in `agent/cortex.txt` and the tests' output in `verifier/`.
