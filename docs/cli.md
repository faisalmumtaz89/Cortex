# Command Line Interface

## Overview

Cortex is a terminal AI coding agent. `cortex` launches the interactive TUI; `cortex -p "..."` runs a single agent turn headlessly. All model and session management happens through slash commands.

## Interactive Mode

```bash
cortex
```

This launches the OpenTUI frontend (single terminal writer), which spawns the Python backend in worker mode (`python -P -m cortex --worker-stdio`; `-P` keeps the project directory off its import path) and talks to it over line-delimited JSON-RPC 2.0.

If the OpenTUI sidecar is unavailable in a source checkout, run `./install.sh` at the repository root to build and install it.

Type a message to run an agent turn with the active model, or a slash command to manage the session.

### Keyboard

| Key | Action |
|---|---|
| `Enter` | Submit input |
| `Shift+Enter` | Insert newline |
| `↑` / `↓`, `Enter` / `Esc` | Choose an answer to a pending permission prompt / reject it |
| `Ctrl+C` | Exit Cortex |

## Headless Mode

```bash
cortex -p "PROMPT" [--model <selector>] [--full-auto]
```

Runs one agent turn through the same worker wiring the TUI uses:

- The assistant reply streams to **stdout**; tool activity and errors go to **stderr**, so stdout stays pipeable.
- `--model` accepts the same `provider:model` selectors as `/model`.
- Permission policy: reads (`read_file`, `list_dir`, `search`) are allowed; `edit_file`, `write_file`, and `bash` are denied unless `--full-auto` is passed (there is no interactive prompt). Persisted rules from `~/.cortex/tool_permissions.yaml` still apply.
- Exit codes: `0` success, `1` turn error, `2` setup error (e.g. model selection failed).

Examples:

```bash
cortex -p "summarize what cortex/app/headless.py does"
cortex -p "rename the helper in src/utils.py and update callers" --full-auto
cortex -p "review this diff for bugs" --model openai:gpt-5.1
```

## Slash Commands

| Command | Description |
|---|---|
| `/help` | List available commands |
| `/status` | Active model and the model that answered the last turn |
| `/model [provider:model]` | Pick a model interactively, or switch by `provider:model` |
| `/login <provider> [api_key]` | Manage OpenAI, Anthropic, Azure, and OpenAI-compatible credentials |
| `/update [cortex]` | Show installed vs latest version, or update Cortex |
| `/clear` | Clear conversation history |
| `/save` | Save the conversation as JSON |
| `/quit` or `/exit` | Exit Cortex |
| `/<skill> [request]` | Run a skill on a request (see [Skills](#skills)) |

### `/model` — switch models

```bash
/model                                  # open the interactive picker (↑↓ + Enter, Esc cancels)
/model openai:gpt-5.1
/model anthropic:claude-sonnet-4-5
/model openai-compatible:deepseek/deepseek-v4.1-flash
/model list                             # plain text list (headless/worker fallback)
```

### `/login` — credentials

```bash
/login openai <api_key>       # store an OpenAI key
/login anthropic <api_key>    # store an Anthropic key
/login azure <api_key>        # store an Azure OpenAI key
/login openai-compatible <api_key>  # store a key for an OpenAI-compatible endpoint
/login openai                 # show auth status for a provider
```

`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `AZURE_OPENAI_API_KEY` environment
variables are used as fallbacks. Azure additionally requires the resource
endpoint via `AZURE_OPENAI_ENDPOINT` (or `cloud_azure_endpoint` in
`~/.cortex/config.yaml`); Azure model ids are your deployment names, selected as
`/model azure:<deployment>` (e.g. `azure:gpt-5.5`).

Any other endpoint that serves the OpenAI Chat Completions API (DeepSeek,
OpenRouter, vLLM, and similar) is reached through the `openai-compatible`
provider. Set its base URL via `OPENAI_COMPATIBLE_BASE_URL` (or
`cloud_openai_compatible_base_url` in `~/.cortex/config.yaml`), provide the key with
`/login openai-compatible <api_key>` or `OPENAI_COMPATIBLE_API_KEY`, and select
the model as `/model openai-compatible:<model>`. The reasoning such models stream
(`reasoning_content` or `reasoning`) is sent back with the reply step it belongs
to, in the same field, for the rest of the conversation: the model continues
from it at each tool step, and DeepSeek requires it whenever tools are offered.


## Agent Tools and Permissions

With tools enabled (the default: `tools_enabled: true`, `tools_profile: full`), the model can call:

- `read_file`, `list_dir`, `search` — read-only, auto-allowed. `read_file` shows each line as `<number>: <text>`, at most 2000 lines and 100 KB per call, with lines longer than 2000 characters cut; when lines are left out, it ends with a note saying how to read on.
- `edit_file` (exact string replacement), `write_file` — prompt for permission. `edit_file` matches text as `read_file` shows it and changes only the text it replaces: the rest of the file keeps its line breaks (`\r\n`, `\n` or `\r`), and its UTF-8 byte order mark. Content `write_file` writes without `\r` gets the line break the old file used most; to change a file's line breaks, use `bash`. A changed file keeps its mode, owner, group, extended attributes and ACLs, and is replaced in one step, so it never holds part of a write. Where that cannot be done keeping all of these (a file with hard links, one owned by another user, one whose attributes cannot be copied, a file mounted on its own, or one in a directory that does not allow new files), the file is written in place instead, which can leave it cut short if the disk fills or the machine stops mid-write. Read-only files are refused, unless Cortex runs as root.
- `bash` — prompts for permission. A command runs with no input (it cannot read from the terminal), with stderr merged into stdout (programs that buffer their output when it is not a terminal may still print out of order), and in your environment minus the provider API keys Cortex uses (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`) and all `CORTEX_*` variables. That keeps those keys out of commands' output; it is not a sandbox, and a project whose own tests need one of those keys has to get it another way. The call returns when the command exits; a background process it started keeps running while Cortex does, and its later output is discarded. At its timeout (2 minutes by default, at most 10), or when you press `Esc`, the command and everything in its process group get SIGTERM, then SIGKILL 3 seconds later; `Esc` also ends the turn. Of long output, the first 20 KB and the last 230 KB are kept, with a note of how many bytes were left out; carriage returns become line breaks.

File tools work only inside the directory Cortex was started in, and `bash` runs there or in a directory below it; `bash` commands are not otherwise sandboxed. When permission is needed, the TUI shows an arrow menu (↑↓ to choose, Enter to confirm):

- **Allow once** — allows this one call; the next call asks again
- **Allow always** — allows this exact command, run in the same directory (for `bash`), or this file (for edits) from now on, only in the directory Cortex was started in; saved in `~/.cortex/tool_permissions.yaml`. A call with no command or file to name is allowed this once only.
- **Reject** (or `Esc`) — the model continues without the tool result

The prompt shows a command on one line: backslashes, line breaks, tabs and invisible or blank-looking characters appear as escapes such as `\n` or `\u{200b}`, and a run of 8 or more blank or invisible characters as a count. A command too long to show in full keeps its start and end, with a count of the characters not shown, and the prompt then starts on **Reject**.

Profiles restrict the exposed tool set: `off` (none), `read_only`, `edit` (adds `edit_file`/`write_file`), `full` (adds `bash`). See `docs/configuration.md` for the `tools_*` keys.

## Skills

A skill is a set of instructions for one kind of task, in a folder holding a
`SKILL.md`:

```markdown
---
name: map-and-plan
description: Use when the user asks how code works or wants a change planned before it is made.
allowed-tools: [read_file, list_dir, search]
---
Step-by-step instructions the model follows...
```

`name` must match the folder's name (lowercase letters, digits and hyphens);
`description` (at most 400 characters) says when the skill applies; the optional
`allowed-tools` lists the only tools the skill uses. Other files in the folder,
such as examples, are read only when the instructions call for them.

Cortex finds skills in three places, and a skill in a later one replaces one of
the same name in an earlier one:

1. the skills that ship with Cortex,
2. `~/.cortex/skills/<name>/`, yours, in every project,
3. `.cortex/skills/<name>/` in the project, trusted as its `AGENTS.md` is.

A skill whose `SKILL.md` cannot be read or is not valid (larger than 1 MB, no
front matter, a missing or wrong field) is skipped, and the reason is logged.

The model sees only each skill's name and description. When a request matches
one, it loads the skill with the `skill` tool, which also reads the skill's
other files (only files inside the skill's folder). To run a skill yourself,
type `/<name>` followed by your request, such as `/map-and-plan add a --json
flag to the export command`; skills are listed in the `/` palette and in
`/help`, and a built-in command of the same name wins. In headless mode,
`cortex -p "/<name> request"` does the same.

Once a skill with `allowed-tools` is loaded, the other tools are refused for
the rest of the turn, so a read-only skill cannot change files even if the
model tries. Loading another skill in the same turn can narrow that set but
never widen it; the next message starts without a limit. With tools off
(`tools_profile: off`), skills are neither listed nor run.

## Configuration

Cortex reads an optional `~/.cortex/config.yaml`. Common keys: `temperature`, `max_tokens`, `tools_profile`, `cloud_default_openai_model`. See the [Configuration Guide](configuration.md).
