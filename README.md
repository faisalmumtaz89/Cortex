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

3. Inside Cortex, type `/login` and pick your provider: OpenAI, Anthropic, Azure OpenAI (it also asks for your endpoint), or any OpenAI-compatible endpoint such as DeepSeek, OpenRouter or vLLM (it also asks for the base URL). Then pick a model with `/model`. You can also set the provider's environment variables instead: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY` with `AZURE_OPENAI_ENDPOINT`, or `OPENAI_COMPATIBLE_API_KEY` with `OPENAI_COMPATIBLE_BASE_URL`.

4. Describe what you want done. Cortex reads `AGENTS.md` (or `CLAUDE.md`) from your project, so your conventions travel with the agent.

5. Stay current with `/update` — Cortex checks daily for a new release and tells you when one is available (`/update cortex` installs it; opt out with `auto_update_check: false` in `~/.cortex/config.yaml`).

## How it works

- Reading, searching, and listing files is free; every edit, write, and shell command asks first (headless `-p` runs refuse them unless you pass `--full-auto`).
- In the interactive app, edits render as green/red diffs before they land.
- Every reply is provenance-verified: it must come from the provider you picked — the model shown is the model that answered.
- Skills give the model step-by-step instructions for kinds of task. Cortex ships five (`map-and-plan`, `reproduce-and-fix`, `safe-refactor`, `review-changes`, `commit-and-pr`), and you can add your own in `~/.cortex/skills/` or the project's `.cortex/skills/`. The model loads one when a request matches it, or you run one with `/<skill> request`. See [Skills](docs/cli.md#skills).
- Headless mode for scripts and CI: `cortex -p "fix the failing test" --model openai:gpt-5.5 --full-auto` (the model is remembered for later runs).

## Commands

`/model` · `/login` · `/status` · `/update` · `/clear` · `/save` · `/help` · `/quit` (or `/exit`) · `/<skill> request`

## Documentation

[Installation](docs/installation.md) ([uninstall](docs/installation.md#uninstallation)) · [CLI & tools](docs/cli.md) · [Configuration](docs/configuration.md) · [Conversations](docs/conversation-management.md) · [Troubleshooting](docs/troubleshooting.md) · [Architecture](docs/architecture-runtime.md) · [Protocol debugging](docs/protocol-debugging.md) · [Development](docs/development.md)

## Contributing

Working on Cortex itself: set up a source checkout and run the test suites as [Development](docs/development.md) describes.

## License

MIT — see [LICENSE](LICENSE).
