# Cortex

Cortex is an agentic coding tool that lives in your terminal. It reads, searches, and edits your code, runs commands, and shows every change as a reviewable diff — powered by models from OpenAI, Anthropic, Azure OpenAI, or any OpenAI-compatible endpoint.

Runs on macOS (Apple Silicon or Intel), Linux (x86_64 or arm64), and Windows through WSL. The installer needs only `curl`, and brings its own Python when yours is missing or too old.

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

3. Add a provider key with `/login openai <api_key>` (or `anthropic` / `azure` / `openai-compatible`), or set the provider's environment variable, then pick a model with `/model`.

4. Describe what you want done. Cortex reads `AGENTS.md` (or `CLAUDE.md`) from your project, so your conventions travel with the agent.

5. Stay current with `/update` — Cortex checks daily for a new release and tells you when one is available (`/update cortex` installs it; opt out with `auto_update_check: false`).

## How it works

- Reading, searching, and listing files is free; every edit, write, and shell command asks first.
- Edits render as green/red diffs before they land.
- Every reply is provenance-verified: it must come from the provider you picked — the model shown is the model that answered.
- Skills give the model step-by-step instructions for kinds of task. Cortex ships five (`map-and-plan`, `reproduce-and-fix`, `safe-refactor`, `review-changes`, `commit-and-pr`), and you can add your own in `~/.cortex/skills/` or the project's `.cortex/skills/`. The model loads one when a request matches it, or you run one with `/<skill> request`. See [Skills](docs/cli.md#skills).
- Headless mode for scripts and CI: `cortex -p "fix the failing test" --full-auto`.

## Commands

`/model` · `/login` · `/status` · `/update` · `/clear` · `/save` · `/help` · `/quit`

## Documentation

[Installation](docs/installation.md) · [CLI & tools](docs/cli.md) · [Configuration](docs/configuration.md) · [Conversations](docs/conversation-management.md) · [Troubleshooting](docs/troubleshooting.md) · [Architecture](docs/architecture-runtime.md) · [Protocol debugging](docs/protocol-debugging.md) · [Development](docs/development.md)

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
