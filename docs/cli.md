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
| `↑` / `↓` (empty input) | Recall earlier prompts |
| `Esc` | Close the palette or result panel; otherwise interrupt the running turn |
| `Ctrl+O` | Show or hide tool details |
| `PageUp` / `PageDown` | Scroll the transcript |
| `Ctrl+C` | Exit Cortex at once, without asking |

## Headless Mode

```bash
cortex -p "PROMPT" [--model <selector>] [--full-auto]
```

Runs one agent turn through the same worker wiring the TUI uses:

- The assistant reply streams to **stdout**; tool activity and errors go to **stderr**, so stdout stays pipeable.
- `--model` accepts the same `provider:model` selectors as `/model`.
- Permission policy: reads (`read_file`, `list_dir`, `search`) are allowed; `edit_file`, `write_file`, and `bash` are denied unless `--full-auto` is passed (there is no interactive prompt): the model is told so, along with the commands your saved rules allow, stderr shows `[tool] bash refused (needs --full-auto)`, and the run ends with a note saying how many calls were refused. Persisted rules from `~/.cortex/tool_permissions.yaml` still apply, so a command you allowed with "Allow always" in the app runs without `--full-auto` (see `tool_permissions.yaml` in the [Configuration Guide](configuration.md)).
- Exit codes: `0` success, `1` turn error (the provider refused the request, for example an unknown model name; the reply was cut off; ...), `2` setup error (no model selected, or model selection failed). A turn whose edits or commands were refused still exits `0`; check stderr for the refusal note.
- Without `--model`, the model last selected (in the TUI or by an earlier `--model`) is used; the first run needs `--model`. If that model cannot be used now (most often: its key is not set in this terminal), the run says why and exits `2`, and the model stays selected for the next run. A `--model` the provider refuses outright (an unknown model name, for example) is not remembered. `-p` is also available as `--print`.
- `-p` needs a prompt: an empty one, or `--model`/`--full-auto` without `-p`, is a usage error (exit `2`) rather than starting the interactive app.
- Each run is a new conversation: it does not see earlier `-p` runs, so say everything the task needs in the prompt.
- Text the model writes after a tool call starts a new paragraph.
- Slash commands are not run in headless mode, except `/<skill> request`. A prompt that starts with a command of the interactive app (`/model`, `/login`, `/update`, ...) is refused with exit code `2` and not sent to the model; any other text, `/` or not, is sent to the model as written.

Examples:

```bash
cortex -p "summarize what src/main.py does" --model openai:gpt-5.5   # the first run picks the model
cortex -p "rename the helper in src/utils.py and update callers" --full-auto
cortex -p "review this diff for bugs" --model openai:gpt-5.5
```

## Slash Commands

| Command | Description |
|---|---|
| `/help` | List available commands |
| `/status` | Active model and the model that answered the last turn |
| `/model [provider:model]` | Pick a model interactively, or switch by `provider:model` |
| `/login <provider> [api_key] [url]` | Manage OpenAI, Anthropic, Azure, and OpenAI-compatible credentials (Azure and OpenAI-compatible also take the URL) |
| `/update [cortex]` | Show installed vs latest version, or update Cortex |
| `/clear` | Clear conversation history |
| `/save` | Save the conversation as JSON |
| `/quit` or `/exit` | Exit Cortex |
| `/<skill> [request]` | Run a skill on a request (see [Skills](#skills)) |

### `/model` — switch models

```bash
/model                                  # open the interactive picker (↑↓ + Enter, Esc cancels)
/model openai:gpt-5.5
/model anthropic:claude-sonnet-5
/model openai-compatible:deepseek/deepseek-v4.1-flash
/model list                             # plain text list
```

### `/login` — credentials

```bash
/login                        # pick a provider; Cortex then asks for its key (and URL)
/login openai <api_key>       # store an OpenAI key
/login anthropic <api_key>    # store an Anthropic key
/login azure <api_key> <endpoint>                # Azure OpenAI key and resource endpoint
/login openai-compatible <api_key> <base_url>    # key and base URL of an OpenAI-compatible endpoint
/login openai                 # show auth status for a provider
```

`/login` saves the key in the system keyring (service `cortex-llm`), such as the
macOS Keychain. Where no keyring works (common on Linux servers, containers and
WSL), it saves the key in `~/.cortex/credentials.json` instead, readable only by
you. The
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY` and
`OPENAI_COMPATIBLE_API_KEY` environment variables take precedence over a saved
key when they are set. Azure additionally requires the resource
endpoint: give it to `/login azure <api_key> <endpoint>`, or set
`AZURE_OPENAI_ENDPOINT` (or `cloud_azure_endpoint` in `~/.cortex/config.yaml`,
which take precedence); Azure model ids are your deployment names, selected as
`/model azure:<deployment>` (e.g. `azure:gpt-5.5`).

Any other endpoint that serves the OpenAI Chat Completions API (DeepSeek,
OpenRouter, vLLM, and similar) is reached through the `openai-compatible`
provider. Give its key and base URL with
`/login openai-compatible <api_key> <base_url>` (for example
`/login openai-compatible sk-... https://api.deepseek.com/v1`), and select the
model as `/model openai-compatible:<model>`; the model you select is listed in
the `/model` picker from then on. Once you are logged in, the picker also has a
row `openai-compatible:… type a model name` that starts that command for you. Outside the app, set `OPENAI_COMPATIBLE_API_KEY`
and `OPENAI_COMPATIBLE_BASE_URL` instead (or `cloud_openai_compatible_base_url` in
`~/.cortex/config.yaml`); these take precedence over what `/login` saved. A base URL in use is also saved in `~/.cortex/state.yaml`, so a later session needs only the key. The reasoning such models stream
(`reasoning_content` or `reasoning`) is sent back with the reply step it belongs
to, in the same field, for the rest of the conversation: the model continues
from it at each tool step, and DeepSeek requires it whenever tools are offered.


## Agent Tools and Permissions

With tools enabled (the default: `tools_enabled: true`, `tools_profile: full`), the model can call:

- `skill` — read-only, auto-allowed; loads a skill's instructions or one of its files (present when any skill exists; see [Skills](#skills)).
- `read_file`, `list_dir`, `search` — read-only, auto-allowed. `read_file` shows each line as `<number>: <text>`, at most 2000 lines and 100 KB per call, with lines longer than 2000 characters cut; when lines are left out, it ends with a note saying how to read on.
- `edit_file` (exact string replacement), `write_file` — prompt for permission. `edit_file` matches text as `read_file` shows it and changes only the text it replaces: the rest of the file keeps its line breaks (`\r\n`, `\n` or `\r`), and its UTF-8 byte order mark. Content `write_file` writes without `\r` gets the line break the old file used most; to change a file's line breaks, use `bash`. A changed file keeps its mode, owner, group, extended attributes and ACLs, and is replaced in one step, so it never holds part of a write. Where that cannot be done keeping all of these (a file with hard links, one owned by another user, one whose attributes cannot be copied, a file mounted on its own, or one in a directory that does not allow new files), the file is written in place instead, which can leave it cut short if the disk fills or the machine stops mid-write. Read-only files are refused, unless Cortex runs as root.
- `bash` — prompts for permission. A command runs with no input (it cannot read from the terminal), with stderr merged into stdout (programs that buffer their output when it is not a terminal may still print out of order), and in your environment minus the provider API keys Cortex uses (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`) and all `CORTEX_*` variables. That keeps those keys out of commands' output; it is not a sandbox, and a project whose own tests need one of those keys has to get it another way. The call returns when the command exits; a background process it started keeps running while Cortex does, and its later output is discarded. At its timeout (2 minutes by default, at most 10), or when you press `Esc`, the command and everything in its process group get SIGTERM, then SIGKILL 3 seconds later; `Esc` also ends the turn. Of long output, the first 20 KB and the last 230 KB are kept, with a note of how many bytes were left out; carriage returns become line breaks.

File tools work only inside the directory Cortex was started in, and `bash` runs there or in a directory below it; `bash` commands are not otherwise sandboxed. When permission is needed, the TUI shows an arrow menu (↑↓ to choose, Enter to confirm):

- **Allow once** — allows this one call; the next call asks again
- **Allow always** — allows this exact command, run in the same directory (for `bash`), or this file (for edits) from now on, only in the directory Cortex was started in; saved in `~/.cortex/tool_permissions.yaml`. A call with no command or file to name is allowed this once only.
- **Reject** (or `Esc`) — the model continues without the tool result

An edit's prompt shows the change it would make, as a diff with line numbers, before anything is written (as many rows as fit the screen, then a count of the rest).

The prompt shows a command on one line: backslashes, line breaks, tabs and invisible or blank-looking characters appear as escapes such as `\n` or `\u{200b}`, and a run of 8 or more blank or invisible characters as a count. A long command fills the rows free on screen; one longer than they hold keeps its start and end, with a count of the characters not shown, and the prompt then starts on **Reject**.

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

Cortex ships five skills:

| Skill | Use it to | Tools |
|---|---|---|
| `map-and-plan` | Map the code a question or change touches, with `path:line` references, and write a step-by-step plan before any edit | read-only: `read_file`, `list_dir`, `search` |
| `reproduce-and-fix` | Find the first real error in a bug report, failing test or CI log, reproduce it with a failing test, fix the code (never the test) and run the full suite | all |
| `safe-refactor` | Restructure code or upgrade dependencies from a passing suite, in small steps, keeping behaviour and public APIs | all |
| `review-changes` | Review a diff for bugs and security flaws, prove each finding by running it, and rank them; it never fixes them | no `edit_file`; `write_file` and `bash` allowed (its instructions use `write_file` only for a review file you ask for) |
| `commit-and-pr` | Check a diff for secrets and debug leftovers, write the commit message and PR description, and run git without force pushes or history rewrites | no `edit_file`; `write_file` and `bash` allowed (its instructions use `write_file` only for a message or PR description file you ask for) |

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
flag to the export command`; skills are listed in the TUI's `/` palette and
`/help`, and a built-in command of the same name wins. In headless mode,
`cortex -p "/<name> request"` does the same.

Once a skill with `allowed-tools` is loaded, the other tools are refused for
the rest of the turn, so a read-only skill such as `map-and-plan` cannot change
files even if the model tries. A skill that keeps `bash` or `write_file` is
held to the rest by its instructions only. Loading another skill in the same turn can narrow that set but
never widen it; the next message starts without a limit. With tools off
(`tools_profile: off`), skills are neither listed nor run.

## Configuration

Cortex reads an optional `~/.cortex/config.yaml`. Common keys: `temperature`, `max_tokens`, `tools_profile`, `cloud_default_openai_model`. See the [Configuration Guide](configuration.md).
