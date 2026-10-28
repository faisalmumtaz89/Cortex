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
| `Enter` | Submit input (a command name typed alone, such as `quit`, runs that command, as the hint under the input says; any longer text is a message) |
| `Shift+Enter` | Insert newline |
| `↑` / `↓`, `Enter` / `Esc` | Choose an answer to a pending permission prompt / reject it. With a message typed in the input, `Enter` queues the message and the prompt stays open; `Esc` still rejects, and the message stays in the input |
| `↑` / `↓` (empty input) | Recall earlier prompts |
| `Esc` | Close the palette or result panel; otherwise interrupt the running turn |
| `Ctrl+O` | Show or hide tool details |
| `PageUp` / `PageDown` | Scroll a permission prompt's call or a command's result when it has more rows than it shows; otherwise the transcript |
| `Home` / `End` | Jump to the first or last rows of such a call or result (with text typed, they move in the text instead) |
| `Ctrl+U` | Delete the typed text before the caret |
| `Ctrl+C` | Exit Cortex at once, without asking |

## Headless Mode

```bash
cortex -p "PROMPT" [--model <selector>] [--full-auto]
```

Runs one agent turn through the same worker wiring the TUI uses:

- The assistant reply streams to **stdout**; tool activity and errors go to **stderr**, so stdout stays pipeable. stderr shows each call (`[tool] bash $ <command as typed>`, `[tool] read_file path=src/app.py`), how it ended, and each edit as a unified diff. A file a command changes (a `sed -i`, a script) is not diffed: run `git diff` after the step to see everything the run changed.
- `--model` accepts the same `provider:model` selectors as `/model`.
- Permission policy: reads (`read_file`, `list_dir`, `search`, `git`) are allowed; `edit_file`, `write_file`, and `bash` are denied unless `--full-auto` is passed (there is no interactive prompt): the model is told so, along with the commands your saved rules allow, stderr shows `[tool] bash refused (needs --full-auto)`, and the run ends with a note saying how many calls were refused. Such a run also starts with `note: this run refuses edits and commands; pass --full-auto to allow them` on stderr, so a run that changed nothing because the model never tried says so too. Persisted rules from `~/.cortex/tool_permissions.yaml` still apply, so a command you allowed with "Allow always" in the app runs without `--full-auto` (see `tool_permissions.yaml` in the [Configuration Guide](configuration.md)).
- stderr ends with the model that answered, as `answered by provider:model (verified)`, after any note. When a refused command is one a saved rule allows for another start directory, a note says so: a rule applies only when Cortex starts in the directory it was saved in (its `repo:`), not in a directory below it.
- Exit codes: `0` success, `1` turn error (the provider refused the request, for example an unknown model name; the reply was cut off; the endpoint could not be reached, with its URL and the reason; ...), `2` setup error (no model selected, or model selection failed), and `128` + the signal when one stops the run (`timeout` sends SIGTERM: `143`), after the command it was running is stopped too; setup errors start with `error:`. A turn whose edits or commands were refused still exits `0`; check stderr for the refusal note.
- Without `--model`, the model last selected (in the TUI or by an earlier `--model`) is used; the first run needs `--model`. A run never falls back to a default model for a key that happens to be set, and a run that fails before it starts remembers no model. If that model cannot be used now (most often: its key is not set in this terminal), the run says why and exits `2`, and the model stays selected for the next run. A `--model` the provider refuses outright (an unknown model name, for example) is not remembered. `-p` is also available as `--print`.
- `-p` needs a prompt, right after it: an empty one, an option in its place (`-p --full-auto "..."`), words outside it, or `--model`/`--full-auto` without `-p` are usage errors (exit `2`) that show the command written correctly, rather than starting the interactive app.
- Each run is a new conversation: it does not see earlier `-p` runs, so say everything the task needs in the prompt.
- stdout gets the answer: when it goes to a pipe or a file, what the model writes before its tool calls ("I'll read the file first.") goes to stderr instead, so a script reads only the final reply. In a terminal everything streams as it is written.
- Input piped to the run is sent after the prompt, so `git diff | cortex -p "summarize this diff"` works (up to 512 KB), and stderr says how much was sent. `cortex -p -` reads the whole prompt from stdin. A stdin left open with nothing in it is ignored after 3 seconds, with a note on stderr; run with `< /dev/null` to skip that wait. In a loop that reads its own input, give each run `< /dev/null`, or the first run takes the rest of the loop's input: `ls src/*.js | while read -r f; do cortex -p "what does $f do" < /dev/null; done`.
- Text the model writes after a tool call starts a new paragraph.
- Slash commands are not run in headless mode, except `/<skill> request`. A prompt that starts with a command of the interactive app (`/model`, `/login`, `/update`, ...) is refused with exit code `2` and not sent to the model (for `/login`, the error names `cortex login`); any other text, `/` or not, is sent to the model as written. A `/name` that is no skill (`/review` for `/review-changes`) is sent too, with a note on stderr that lists the skills.

### In CI

A GitHub Actions job that installs a pinned version and asks about the change:

```yaml
- name: Install Cortex
  run: curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh | CORTEX_NO_MODIFY_PATH=1 bash -s -- 2.0.10
- name: Summarize the change
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    CORTEX_AUTO_UPDATE_CHECK: "false"   # no release check on the runner
    CORTEX_AUTO_SAVE: "false"           # no conversation history kept
    CORTEX_CLOUD_MAX_RETRIES: "4"       # ride out a provider's brief outage
  run: git diff origin/main...HEAD | ~/.local/bin/cortex -p "summarize this change for a reviewer" --model openai:gpt-5.5
```

- Pin the version, so a new release changes nothing until you move the pin.
  With `CORTEX_NO_MODIFY_PATH=1` no shell profile changes, and Cortex runs as
  `~/.local/bin/cortex`.
- Pass `--model` on every run: a fresh runner has no remembered model.
- A request that fails to connect, times out, is rate limited (429) or gets
  a server error (5xx) is sent again, up to `cloud_max_retries` times, before
  any of its reply has arrived; a run that still fails exits `1`, with the
  reason on stderr. Running the step once more is reasonable then; a `2` (no
  key, no model) fails the same way again.
- Exit `0` means the turn finished, not that the task succeeded, so check the
  result yourself (re-run the tests, look at `git diff`); with `--full-auto`
  the model may change any file in the repository.

### Saving a key without the interactive app

```bash
printf %s "$OPENAI_API_KEY" | cortex login openai          # the key from stdin
cortex login openai-compatible https://llm.example.com/v1    # in a terminal: asks for the key
```

`cortex login PROVIDER [URL]` saves the key where `/login` does (the system keyring, else `~/.cortex/credentials.json`, readable only by you), so later `cortex -p` runs on that machine find it. It reads the key from stdin, or asks for it without showing it when run in a terminal; it never takes the key as an argument, which would keep it in shell history and process listings. `azure` takes its endpoint and `openai-compatible` its base URL, unless one is saved or set already. On a CI runner, a secret in the `*_API_KEY` variable is usually simpler.

### A pre-commit check: the tests decide, the review explains

A model's verdict on code it has not run is an opinion, and can call a correct
change a bug. So let the tests stop a commit, and have Cortex review it with the
`review-changes` skill, which reports a finding as proven only by a run that
shows it. A git pre-commit hook:

```bash
#!/bin/sh
# .git/hooks/pre-commit — the tests decide; the review says what may be wrong.
git diff --cached --quiet && exit 0      # nothing staged: nothing to check
dir=$(git rev-parse --git-dir)
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider >"$dir/cortex-tests.log" 2>&1
tests=$?
if [ "$tests" -eq 0 ]; then result=passed; else result=failed; fi
# --model names the model on every run, as on a fresh CI runner, which remembers none.
git diff --cached | cortex -p "/review-changes the staged diff below. The tests ran on it and $result:
$(tail -n 30 "$dir/cortex-tests.log")" --model openai:gpt-5.5 2>>"$dir/cortex-review.log" \
  || echo "pre-commit: the review did not run; see $dir/cortex-review.log" >&2
if [ "$tests" -ne 0 ]; then
  echo "pre-commit: the tests fail; see $dir/cortex-tests.log (commit with --no-verify to skip)" >&2
  exit 1
fi
```

Put your project's test command in place of `python -m pytest ...`. The review
runs nothing itself (a run without `--full-auto` refuses commands), so it works
from the diff, the code it reads and the test output it is given: on a change
that breaks a test it names the line that does, and a bug no test covers (an
off-by-one in a new function, say) it still reports, as unverified, with the
command that would show it, while the commit goes through. Read what it says;
only the tests stop the commit. A review that cannot run (no key, no model)
says so and does not stop it either. The tests run on your working tree,
unstaged changes included: run `git stash --keep-index` first to test exactly
what you commit.

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
| `/status` | The Cortex version, the directory it works in, the active model, the model that answered the last turn, and the settings `config.yaml` and `CORTEX_*` variables changed (each with where it comes from) |
| `/model [provider:model]` | Pick a model interactively, or switch by `provider:model` |
| `/login <provider> [api_key] [url]` | Manage OpenAI, Anthropic, Azure, and OpenAI-compatible credentials (Azure and OpenAI-compatible also take the URL) |
| `/update [cortex]` | Show the installed version and any newer one, with a link to its release notes, or update Cortex (the download is checked against its published checksum before anything is installed) |
| `/clear` | Clear conversation history |
| `/save` | Save the conversation as JSON |
| `/quit` or `/exit` | Exit Cortex |
| `/<skill> [request]` | Run a skill on a request (see [Skills](#skills)) |

A command's result, such as the list `/help` shows, opens in a panel above the
input, and `Esc` closes it. When it has more rows than the screen leaves it, a
row under it says which are shown (`rows 1–7 of 27 · PgUp/PgDn, Home/End`), and
those keys scroll it.

### `/model` — switch models

```text
/model                                  # open the interactive picker (↑↓ + Enter, Esc cancels)
/model openai:gpt-5.5
/model anthropic:claude-sonnet-5
/model openai-compatible:deepseek/deepseek-v4.1-flash
/model list                             # plain text list
```

### `/login` — credentials

```text
/login                        # pick a provider; Cortex then asks for its key (and URL)
/login openai <api_key>       # store an OpenAI key
/login anthropic <api_key>    # store an Anthropic key
/login azure <api_key> <endpoint>                # Azure OpenAI key and resource endpoint
/login openai-compatible <api_key> <base_url>    # key and base URL of an OpenAI-compatible endpoint
/login openai                 # show auth status for a provider
```

While you type the key, the input shows a `•` for each of its characters, and
the result shows it masked: the key itself is never drawn on screen, where a
recording, a screen reader or a copy of the terminal would keep it.

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
`~/.cortex/config.yaml`); these take precedence over what `/login` saved. A base URL in use is also saved in `~/.cortex/state.yaml`, so a later session needs only the key. The `openai` and `anthropic` providers always use their official hosts: `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL`, which their SDKs read, are ignored (the startup notice names one that is set), so a variable left in an environment cannot send their keys elsewhere. Any other endpoint is `openai-compatible`. The reasoning such models stream
(`reasoning_content` or `reasoning`) is sent back with the reply step it belongs
to, in the same field, for the rest of the conversation: the model continues
from it at each tool step, and DeepSeek requires it whenever tools are offered.


## Agent Tools and Permissions

With tools enabled (the default: `tools_enabled: true`, `tools_profile: full`), the model can call:

- `skill` — read-only, auto-allowed; loads a skill's instructions or one of its files (present when any skill exists; see [Skills](#skills)).
- `read_file`, `list_dir`, `search` — read-only, auto-allowed. `read_file` shows each line as `<number>: <text>`, at most 2000 lines and 100 KB per call, with lines longer than 2000 characters cut; when lines are left out, it ends with a note saying how to read on.
- `git` — read-only, auto-allowed; the repository's state: `status`, `diff` (the working tree, the staged changes, or between commits), `log`, `show` (a commit, or a file at one, as `commit:path`) and `merge-base`, each with a few fields (a revision, paths, `stat`, `staged`, `patch`) rather than free-form options. Git runs with the programs a repository's own `.git/config` can name for these commands switched off (fsmonitor, hooks, filters, external diffs, textconv, pager, signature checks), so a downloaded project cannot run code through it; your global git config, git-lfs's filter included, still applies. Anything that changes the repository goes through `bash`.
- `edit_file` (exact string replacement), `write_file` — prompt for permission. `edit_file` matches text as `read_file` shows it and changes only the text it replaces: the rest of the file keeps its line breaks (`\r\n`, `\n` or `\r`), and its UTF-8 byte order mark. Content `write_file` writes without `\r` gets the line break the old file used most; to change a file's line breaks, use `bash`. A changed file keeps its mode, owner, group, extended attributes and ACLs, and is replaced in one step, so it never holds part of a write. Where that cannot be done keeping all of these (a file with hard links, one owned by another user, one whose attributes cannot be copied, a file mounted on its own, or one in a directory that does not allow new files), the file is written in place instead, which can leave it cut short if the disk fills or the machine stops mid-write. Read-only files are refused, unless Cortex runs as root.
- `bash` — prompts for permission. A command runs with no input (it cannot read from the terminal), with stderr merged into stdout (programs that buffer their output when it is not a terminal may still print out of order), and in your environment minus the provider API keys Cortex uses (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`) and all `CORTEX_*` variables. That keeps those keys out of commands' output; it is not a sandbox, and a project whose own tests need one of those keys has to get it another way. The call returns when the command exits; a background process it started keeps running while Cortex does, and its later output is discarded. At its timeout (2 minutes by default, at most 10), or when you press `Esc`, the command and everything in its process group get SIGTERM, then SIGKILL 3 seconds later; `Esc` also ends the turn. Of long output, the first 20 KB and the last 230 KB are kept, with a note of how many bytes were left out; carriage returns become line breaks.

File tools work only inside the directory Cortex was started in (a call for a file elsewhere is refused without a prompt), and `bash` runs there or in a directory below it; `bash` commands are not otherwise sandboxed. When permission is needed, the TUI shows an arrow menu (↑↓ to choose, Enter to confirm):

- **Allow once** — allows this one call; the next call asks again
- **Allow always** — allows this exact command, run in the same directory (for `bash`), or this file (for edits) from now on, only in the directory Cortex was started in; saved in `~/.cortex/tool_permissions.yaml`, as the call's row then says (or that it could not be saved, and why: the rule then holds until Cortex exits). A call with no command or file to name is allowed this once only, and its row says so.
- **Reject** (or `Esc`) — the model continues without the tool result

An edit's prompt shows the file and the size of the change (`+3 -2`), and the change itself as a diff with line numbers before anything is written, starting just before the first changed line; a long line wraps under its code. Invisible and control characters in the new text show as markers such as `⟨U+202E⟩`, a tab as `⟨tab⟩`.

The prompt shows a command as typed, backslashes included, each of its lines starting a row and ending in ` ↵` when another follows. A row that only continues a long line has no ` ↵` above it. Tabs, carriage returns and invisible or blank-looking characters show as markers (`⟨tab⟩`, `⟨CR⟩`, `⟨U+200B⟩`), and a run of 8 or more of them as a count (`⟨12 blank or invisible characters⟩`), except spaces that indent a line. A `⟨`, `⟩` or `↵` in the command itself shows as a marker too, so it cannot pass for one the prompt drew.

The call takes the rows the screen has free but 3, which the transcript keeps so what led to the prompt stays in view. A command's `Run command:` label, and an edit's file with the size of its change, stay in view however far the call scrolls. When the call has more rows than it shows, a row under it says which are shown (`rows 9–16 of 61 · PgUp/PgDn, Home/End`), PgUp/PgDn page through it, Home/End jump to either end, and the prompt starts on **Reject**: Enter approves only a call that was on screen whole, and seeing the rest does not change the choice. So does a diff Cortex had to cut (a line over 2000 characters, or more than 400 rows), and an edit of a file too large to compare. While the call fits, PgUp/PgDn scroll the transcript.

The pickers, the slash command list and a command's result share the screen the same way: each shows the rows the screen leaves it, with the rest counted (`+5 more`) or scrolled, from 80x24 up.

Profiles restrict the exposed tool set: `off` (none), `read_only`, `edit` (adds `edit_file`/`write_file`), `full` (adds `bash`). See the [Configuration Guide](configuration.md) for the `tools_*` keys.

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
| `map-and-plan` | Map the code a question or change touches, with `path:line` references, and write a step-by-step plan before any edit | read-only: `read_file`, `list_dir`, `search`, `git` |
| `reproduce-and-fix` | Find the first real error in a bug report, failing test or CI log, reproduce it with a failing test, fix the code (never the test) and run the full suite | all |
| `safe-refactor` | Restructure code or upgrade dependencies from a passing suite, in small steps, keeping behaviour and public APIs | all |
| `review-changes` | Review a diff for bugs and security flaws, prove each finding by running it, and rank them; it never fixes them. Its reproductions go in a temporary folder it makes outside the project, under `/tmp` (with `mktemp -d`). In `cortex -p` without `--full-auto` it can run nothing, so it lists its findings as **Unverified** | no `edit_file`; `write_file` and `bash` allowed (its instructions use `write_file` only for a review file you ask for) |
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
