# Changelog

What changed in each release of Cortex, newest first. The release workflow
publishes a version's section as its release notes, and stops when a version
has none. Releases before 2.0.9 are described on their GitHub release pages.

## 2.0.16

- `/clear` typed while a reply is running is refused, and says how to
  stop the reply. Before, the screen cleared while the model went on
  working unseen, and the rest of its reply was never shown.
- A command run while a reply is running, such as `/status`, leaves the
  reply shown as running: Esc closes the command's result, and the next
  Esc stops the reply. Before, the "Working…" line went away and Esc did
  nothing until the reply ended.
- With details open (Ctrl+O), a command's row shows the whole text its
  permission was matched against, as the prompt showed it: line by line,
  after the `cd` line for a directory the call named. The row itself is a
  summary, cut at the first pipe, so two commands that "Allow always"
  tells apart could look the same. It is there once the command has run
  or been refused, up to 40 lines, then `⟨N more lines⟩`; a tool's output
  with details open ends the same way, where it ended in
  `… (N more lines)`, which output could imitate.
- The model is sent a tool's result without what is for the app's row of
  the call: an edit's diff (which OpenAI, Azure and Anthropic models were
  sent, up to 400 rows a call) and the line that says what "Allow always"
  saved (which every model was sent).
- A reply from an OpenAI-compatible model that stops at the output limit
  (`max_tokens`) is continued: the model is told its reply was cut off
  and goes on from where it stopped, with what it had written kept. A
  tool call that was cut off does not run, and the model is told to make
  it again. Before, one such reply ended the turn, or a `-p` run; now the
  turn ends with that error only when three replies in a row are cut off.
  With OpenAI, Azure and Anthropic models the first cut-off reply still
  ends the turn.
- Cortex writes no history of conversations: `conversations.db`, which
  every message was written to and nothing read, is gone from new
  installs. One an earlier version left holds what you typed and what
  commands printed; it is not touched, and can be deleted. `auto_save`,
  which could turn the writing off, has no effect and is named at
  startup if a config file still sets it; a `CORTEX_AUTO_SAVE` variable
  is ignored. `/save` works as before.
- A command the agent is running ends when Cortex does, also when the
  app is killed without a signal reaching its worker. Before, the
  command ran on.
- The installer says why it stops on a system with musl libc, such as
  Alpine. Before, it ended there with no message.
- What a background update writes after `/clear` is shown. It was left
  out when it continued a message begun before the `/clear`.

## 2.0.15

- A long line in a command's result shown while `/help` is still open
  wraps to the panel's left edge; it was indented like the help list's.
  `/help` opened while `/update cortex` runs stays open when the update
  ends.
- "Allow always" says what it covers: this exact command, or this file. It
  says so beside the choice, where the screen has room, and on the call's
  row once saved. What it covers has not changed.
- `/save` says what it saves: a copy to keep, which Cortex cannot reopen,
  each start being a new conversation. Its result says so before the
  file's path, and so does its line in `/help`. The line at the start
  that named the model as "Restored" now reads "Using ..., the model you
  last chose".
- A tool call's row in the transcript shows control characters,
  text-direction overrides, zero-width characters and blanks that are not
  spaces as markers such as `⟨U+001B⟩`, the ones the permission prompt
  uses; before, they reached the terminal as written. This holds for the name of
  a tool that is none of Cortex's, for what a call names (a command, path
  or search, where a tab or a line break reads as a space), and for its
  result, its error, its output with details open, and the path and lines
  of an edit. An emoji made of several characters shows the invisible ones
  that join it.
- A correction to the 2.0.14 notes, which said an error stays on screen
  while you run commands. That holds for an error shown above the input,
  such as that of a reply that failed. A command that fails shows its
  message as its result, and the next command's result replaces it.
- A start no longer reads the conversation history: Cortex cannot reopen a
  conversation, and loaded up to 100 earlier ones at every start for
  nothing. `max_conversation_history`, which set that number, has no
  effect and is named at startup if a config file still sets it. The
  app's protocol no longer accepts a conversation to resume.
- Selecting a model by a name the picker did not list says "Saved as your
  model once it answers". A name saved that way is removed when the
  provider answers that it does not know the model (a 404 that says so):
  the error says it was removed, in the app and in a `-p` run, and the
  next start does not restore it. It is saved again once it answers.
- A new setting, `reasoning_effort` (`CORTEX_REASONING_EFFORT` for one
  run), asks the model for that much reasoning: one of the provider's
  levels, such as `low` or `high`. It is sent as `reasoning_effort` to an
  OpenAI-compatible endpoint, as `reasoning.effort` to OpenAI and Azure,
  and as `output_config.effort` to Anthropic. Unset, nothing is sent, as
  before. `/status` lists it. When a provider answers 400 or 422 to a
  request that carried it, the error says what was asked and where it is
  set; the request is not sent again without it.
- A `CORTEX_*` variable for a setting that is text, such as a model's
  name, is taken as written: `4.10` was read as the number 4.1 and
  ignored.
- Cortex needs `openai` 1.66.0 and `anthropic` 0.78.0 or later, the first
  versions with the calls it makes; the installer provides them.

## 2.0.14

- A password or key in an Azure endpoint or OpenAI-compatible base URL no
  longer reaches the log (`~/.cortex/cortex.log`, or your `log_file`), or
  the terminal with `debug_mode` on. 2.0.13 masked it in Cortex's own
  messages only, not in the requests its libraries log. If your URL
  carried one, an earlier log or CI output captured with `debug_mode` may
  hold it: delete the log and replace that password or key.
- The log, the conversation history and conversations saved with `/save`
  are created readable only by you. A copy or backup of `~/.cortex`, or a
  `log_file` set elsewhere, no longer leaves a new one readable by other
  users. A path that is a link is left as you set it. An existing log or
  history in `~/.cortex` is tightened when Cortex next opens it, unless
  it has a second hard link. Conversations saved earlier keep their mode:
  `chmod 600 ~/.cortex/conversations/*.json` tightens those in the
  default folder.
- SIGTERM, SIGHUP or SIGINT ends the Cortex process behind the app at
  once. One delivered to a thread other than the main one used to be
  ignored until Cortex's input closed, so the process could live on.
- An error stays on screen while you run `/status`, `/help` and the other
  commands, so what it says to do next can still be read. It goes when you
  send the next message, choose a model, save a key, or run `/clear`.
- Ctrl+O shows the folded steps and long diffs above a permission prompt
  while it waits. The key did nothing there, though the screen named it.
- `cortex -p "/update"` says how to update without the app: run the
  installer again.
- A 403 or 404 error whose message from the server already ends with a
  full stop no longer reads "... not found.. Check the model name".
- The CI example in the CLI docs shows where the version to pin goes; it
  named 2.0.10.

## 2.0.13

- The CLI docs no longer imply that `timeout` reports exit code `143`:
  Cortex exits `143` on SIGTERM, but `timeout` itself exits `124` unless
  given `--preserve-status`.
- `/status` shows the Azure endpoint or OpenAI-compatible base URL the
  active model uses, and where it is set: its environment variable,
  `config.yaml` or its `CORTEX_CLOUD_*` variable, or saved in
  `~/.cortex/state.yaml`. A `/login` or `cortex login` that saves only a
  key names its provider's URL the same way.
- A run stopped by a signal while a command runs (`timeout`, a cancelled
  CI job) exits as soon as the command has ended, instead of always waiting
  3 seconds: the 3 seconds before SIGKILL are now only for a command that
  keeps running.
- A model picked by typing a name the `/model` picker does not list is
  saved only after it answers, so a mistyped name is not restored next
  session or kept in the picker. Until then, in the app, a failed reply
  says the model is not saved and how to choose another.
- An error the URL may explain (an Azure endpoint or OpenAI-compatible base
  URL that could not be reached, timed out, answered 401, 403 or 404, or
  does not start with `https://` or `http://`) says where that URL is set,
  as `/status` does. For a saved one, which may be left from an earlier
  run, it gives the `/login` and `cortex login` commands that change it.
  The "must start with" error names only the place the URL is set; it
  used to name both the variable and `/login`, whichever that place was.
- A password in an Azure endpoint or OpenAI-compatible base URL
  (`user:password@`) is masked also when the URL does not start with
  `https://` or `http://` and a host, or cannot be parsed. Before, the
  "must start with" error, `/login`, `cortex login` and the settings
  `/status` lists printed it, CI logs included.

## 2.0.12

- Selecting a model saves the model, not the Azure endpoint or
  OpenAI-compatible base URL in effect: only a URL given to `/login` or
  `cortex login` is saved. A later run that does not set
  `AZURE_OPENAI_ENDPOINT` or `OPENAI_COMPATIBLE_BASE_URL` does not send
  your key to the URL an earlier run set there; give the URL to `/login` to
  keep it. A URL saved this way before 2.0.12 stays in
  `~/.cortex/state.yaml` until `/login` or `cortex login` replaces it or
  you delete its line. The `/login` picker shows "URL saved, no key" for a
  provider only when a URL is saved for it and no key is.

## 2.0.11

- A permission prompt that stops showing its call whole while it waits (the
  screen shrank, an error took rows) moves to Reject. On a screen too small
  to show a call with its choices, the prompt starts on Reject.
- While a permission prompt waits, the `/model` and `/login` pickers and a
  command's hint panel are not drawn: the prompt takes Enter and Esc. A
  picker opened then shows once the prompt is answered.
- A key in a URL's query (`?api_key=`, `?code=`, a signature) is masked
  wherever Cortex shows the URL, like a `user:password@` before the host.
- An error that wraps to several rows keeps the input in view on a full
  screen, and leaves room for a picker or the slash command list opened
  under it.
- A long line in a reply's code block goes on in rows that start with
  `↪`, every character kept, wide characters and emoji included; tabs
  show as 4 spaces.
- `cortex -p` with stdout to a pipe or a file: a reply an error or the
  output limit cut off goes to stderr with the reason, so stdout is empty
  when the turn fails.

## 2.0.10

The terminal app
- Text is cut into rows by Cortex, at spaces: a reply, notice or message
  that wraps keeps every character (code a reply quoted could lose the
  character where a row broke).
- On a small screen nothing is drawn over anything else: the pickers, the
  slash command list, a command's result and a permission prompt take the
  rows the screen leaves them, keeping 3 for the transcript.
- `/help`, `/status` and the other results scroll with PgUp/PgDn and
  Home/End when taller than their panel.
- A permission prompt shows a command as typed, backslashes included, each
  line ending in ` ↵` when another follows; tabs and invisible characters
  show as markers such as `⟨tab⟩` and `⟨U+200B⟩`. `Run command:` stays in
  view while the command scrolls, Home/End jump to either end, and an edit
  opens on the line before its change.
- "Allow always" says on the call's row what it saved and where, or that it
  could not be saved and why.
- The `/login` picker starts on the provider in use or logged in.
- With text typed, Home and End move in it rather than scroll.

Headless runs (`cortex -p`)
- A run uses only a model chosen with `--model`, now or by an earlier run
  or in the app; it never falls back to a default for a key that happens to
  be set, and a run that fails remembers no model.
- The `openai` and `anthropic` providers always use their official hosts:
  `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL`, which their SDKs read, are
  ignored and named in the startup notice. Any other endpoint is
  `openai-compatible`.
- A run stopped by SIGTERM, SIGHUP or SIGINT stops the command it is running
  and exits 128 + the signal; the app's worker does the same.
- A run without `--full-auto` says on stderr, before it starts, that it
  refuses edits and commands.
- A connection that fails names the URL, with credentials in it masked, and
  the reason.

Updates and settings
- The update notice and `/update` link to the release's notes, and
  `/update cortex` says the download's checksum was verified. The first
  session after an update says what version it runs and links to its notes.
- `/status` lists the settings `config.yaml` and `CORTEX_*` variables
  changed, and where each comes from.
- `state.yaml` is written readable only by you, in one step.

Docs
- A pre-commit check where the tests decide and the review explains,
  checked against correct and buggy changes; a recipe for CI.

## 2.0.9

- A permission prompt shows all of a call: it scrolls with PgUp/PgDn, starts
  on Reject when the call is not on screen whole, and never draws past the
  screen.
- A `/login` key is never drawn while it is typed.
- Headless errors point at `cortex login` and never repeat a key.
- The installer uses the Python it is told to, or stops and says why.
- The install docs show where an installer variable goes in a pipe.
