# Changelog

What changed in each release of Cortex, newest first. The release workflow
publishes a version's section as its release notes, and stops when a version
has none. Releases before 2.0.9 are described on their GitHub release pages.

## 2.0.12

- Selecting a model saves the model, not the Azure endpoint or
  OpenAI-compatible base URL in effect: only a URL given to `/login` or
  `cortex login` is saved. A later run that does not set
  `AZURE_OPENAI_ENDPOINT` or `OPENAI_COMPATIBLE_BASE_URL` does not send
  your key to the URL an earlier run set there; give the URL to `/login` to
  keep it. A URL saved this way before 2.0.12 stays in
  `~/.cortex/state.yaml` until `/login` or `cortex login` replaces it or
  you delete its line. `/login` shows "URL saved" only for a saved URL.

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
