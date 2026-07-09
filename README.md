# Cortex

![Cortex demo](demo.gif)

Cortex is an agentic coding tool that lives in your terminal. It reads, searches, and edits your code, runs commands, and shows every change as a reviewable diff — powered by local models running GPU-resident on your Mac through the [Lumen](https://github.com/faisalmumtaz89/Lumen) inference engine, with optional cloud models from OpenAI, Anthropic, and Azure OpenAI.

Requires an Apple Silicon Mac (M1–M4), macOS 13.3+, Python 3.11+, and Xcode Command Line Tools.

## Get started

1. Install Cortex:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash
   ```

2. Navigate to your project and start Cortex:

   ```bash
   cd your-project
   cortex
   ```

3. Pick a model with `/model`:

   - **Local** — Qwen3.5 / 3.6 models served by Lumen. Selecting one downloads and loads it automatically; Cortex manages the server for you.
   - **Cloud** — `/login openai <api_key>` (or `anthropic` / `azure`), or set the provider's environment key.

4. Describe what you want done. Cortex reads `AGENTS.md` (or `CLAUDE.md`) from your project, so your conventions travel with the agent.

5. Stay current with `/update` — Cortex checks daily for new Cortex and Lumen releases and tells you when one is available (`/update lumen` upgrades the local engine in place; opt out with `auto_update_check: false`).

## How it works

- Reading, searching, and listing files is free; every edit, write, and shell command asks first.
- Edits render as green/red diffs before they land.
- Every reply is provenance-verified: local turns must come from Cortex's own Lumen server, cloud turns from the provider you picked — the model shown is the model that answered.
- Headless mode for scripts and CI: `cortex -p "fix the failing test" --full-auto`.

## Commands

`/model` · `/download` · `/login` · `/status` · `/gpu` · `/benchmark` · `/update` · `/clear` · `/save` · `/setup` · `/help` · `/quit`

## Documentation

[Installation](docs/installation.md) · [CLI & tools](docs/cli.md) · [Model management](docs/model-management.md) · [Configuration](docs/configuration.md) · [Troubleshooting](docs/troubleshooting.md) · [Architecture](docs/architecture-runtime.md) · [Development](docs/development.md)

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q          # full suite — must be green before and after any change
./scripts/typecheck.sh              # mypy
cd frontend/cortex-tui && bun run typecheck && bun run build   # TUI sidecar
```

Behavioral changes are gated empirically: e2e suites drive the real worker over JSON-RPC and the real TUI in tmux with a deterministic scripted model, asserting observable effects. If a change cannot be observed through the suite, add the E2E scenario first. See [Development](docs/development.md).

## License

MIT — see [LICENSE](LICENSE).
