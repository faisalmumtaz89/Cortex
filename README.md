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

Cortex runs coding-agent turns in your terminal: it reads, searches, edits, and runs commands inside your repository through a small permissioned tool set. Local-first — models run on your Mac's GPU via MLX (primary) or GGUF (llama.cpp) — with optional cloud models from OpenAI, Anthropic, and Azure OpenAI.

Two processes: an OpenTUI frontend (Bun + SolidJS, `frontend/cortex-tui`) owns rendering; a Python worker (`python -m cortex --worker-stdio`) owns models and tools, speaking JSON-RPC 2.0 over stdio.

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | bash
cortex
```

Inside Cortex:

- `/download mlx-community/Nanbeige4.1-3B-bf16 --load` — fetch and load a local model
- `/login openai <api_key>` (or `anthropic` / `azure`) — use cloud models instead; env keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`) also work
- `/model` — pick a model interactively (↑↓ + Enter), or `/model <name | provider:model>`
- `/help` — every command

Then type what you want done. Cortex loads `AGENTS.md` (or `CLAUDE.md`) from your working directory into its system prompt, so project conventions travel with the agent.

## Agent Tools

Tools are sandboxed to the directory Cortex was started in:

| Tool | Purpose | Permission |
|---|---|---|
| `read_file` | Read a file by optional line range | auto-allowed |
| `list_dir` | List directory contents | auto-allowed |
| `search` | Search file contents (prefers ripgrep) | auto-allowed |
| `edit_file` | Exact string-replacement edit | prompts |
| `write_file` | Create or overwrite a file | prompts |
| `bash` | Run a shell command with timeout | prompts |

Edits render as green/red diffs. When approval is needed, an arrow menu opens: **Allow once** / **Allow always** (persisted to `~/.cortex/tool_permissions.yaml`) / **Reject** — ↑↓ to choose, Enter to confirm, Esc rejects.

Profiles (`tools_profile` in `config.yaml`): `off`, `read_only`, `edit`, `full` (default). Local models call tools through an experimental `<tool_calls>` JSON protocol; cloud models use native tool calling.

## Headless Mode

One agent turn without the TUI — reply on stdout, tool activity on stderr, pipeable:

```bash
cortex -p "explain the retry logic in cortex/cloud/router.py"
cortex -p "fix the failing test" --model anthropic:claude-sonnet-4-5 --full-auto
```

Reads are allowed; edits and shell commands are denied unless `--full-auto`. Exit codes: `0` success, `1` turn error, `2` setup error.

## Platform Support

Local inference is Apple Silicon-only (MLX and llama.cpp-Metal require it); the agent itself — worker, TUI, tools, cloud models — is portable Python + Bun.

| Platform | Support |
|---|---|
| Apple Silicon Mac (M1–M4, macOS 13.3+) | **Full** — local models on the GPU plus cloud models. The installer targets this platform only. |
| Linux | **Cloud-only, from source** — install with `pip install -e .`, then `/login` a provider. Local models are unavailable; no installer. Works (the benchmark harness runs Cortex this way in Linux containers) but is not officially supported. |
| Windows | Not supported natively (the agent's shell tool and TUI build assume POSIX). WSL2 works — it follows the Linux path above. |
| Intel Mac | Not supported. |

macOS requirements: Python 3.11+, Xcode Command Line Tools, 8GB+ unified memory (16GB+ recommended for larger local models).

## Documentation

- `docs/installation.md` · `docs/cli.md` · `docs/model-management.md` · `docs/configuration.md` · `docs/troubleshooting.md`
- Deeper: `docs/architecture-runtime.md`, `docs/inference-engine.md`, `docs/mlx-acceleration.md`, `docs/development.md`

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q          # full suite — must be green before and after any change
./scripts/typecheck.sh              # mypy
cd frontend/cortex-tui && bun run typecheck && bun run build   # TUI sidecar
```

Behavioral changes are gated empirically: `tests/test_agent_runtime_e2e.py` drives the real worker over JSON-RPC and `tests/test_tui_e2e.py` drives the real TUI in tmux, both with a deterministic scripted model (`CORTEX_SCRIPTED_MODEL`), asserting observable effects. If a change cannot be observed through the suite, add the E2E scenario first. See `docs/development.md`.

## License

MIT License. See `LICENSE`.
