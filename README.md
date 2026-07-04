# Cortex

```
  ██████╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗
 ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝
 ██║     ██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝
 ██║     ██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗
 ╚██████╗╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗
  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

        terminal AI coding agent for Apple Silicon
```

Cortex is an agentic coding tool that lives in your terminal. It reads, searches, and edits your code, runs commands, and shows every change as a reviewable diff — powered by local models running on your Mac's GPU through the [Lumen](https://github.com/faisalmumtaz89/Lumen) inference engine, or by cloud models (OpenAI, Anthropic, Azure OpenAI) when you want them.

![Cortex demo](demo.gif)

**Cortex runs on Apple Silicon Macs. Nothing else.**

## Get started

1. Install Cortex:

```bash
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash
```

2. Navigate to your project and start:

```bash
cd your-project
cortex
```

3. Pick a model with `/model`:
   - **Local** — Lumen-supported models (Qwen3.5 / 3.6 family). Selecting an undownloaded model downloads and loads it automatically; Cortex manages the Lumen server for you (start, load, switch, shutdown).
   - **Cloud** — `/login openai <key>` (or `anthropic` / `azure`), or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`.

Then type what you want done. Cortex loads `AGENTS.md` (or `CLAUDE.md`) from your working directory into its system prompt, so project conventions travel with the agent.

## What you get

- **Agent tools, sandboxed to your project** — `read_file`, `list_dir`, `search` run freely; `edit_file`, `write_file`, `bash` ask first (arrow menu: allow once / always / reject). Edits render as green/red diffs.
- **Local models, fully managed** — Cortex owns the Lumen server lifecycle end to end; local turns use the same native tool-calling loop as cloud.
- **Verified provenance** — every reply is checked against the model you selected (endpoint + reported model); a mismatch rejects the turn. What the UI shows is what actually answered, labeled `local ·` / `cloud ·` everywhere.
- **Headless mode** — `cortex -p "fix the failing test" --full-auto` for scripts and CI; reply on stdout, tools on stderr.

## Commands

`/model` (tabbed local/cloud picker) · `/download` · `/login` · `/status` · `/gpu` · `/benchmark` · `/clear` · `/save` · `/setup` · `/help` · `/quit`

## Documentation

- [Installation](docs/installation.md) · [CLI & tools](docs/cli.md) · [Model management](docs/model-management.md) · [Configuration](docs/configuration.md) · [Troubleshooting](docs/troubleshooting.md)
- Deeper: [Runtime architecture](docs/architecture-runtime.md) · [Development](docs/development.md)

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q          # full suite — must be green before and after any change
./scripts/typecheck.sh              # mypy
cd frontend/cortex-tui && bun run typecheck && bun run build   # TUI sidecar
```

Behavioral changes are gated empirically: e2e suites drive the real worker over JSON-RPC and the real TUI in tmux with a deterministic scripted model, asserting observable effects. If a change cannot be observed through the suite, add the E2E scenario first. See [docs/development.md](docs/development.md).

## License

MIT License. See `LICENSE`.
