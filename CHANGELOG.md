# Changelog

What changed in each release of Cortex, newest first. The release workflow
publishes a version's section as its release notes, and stops when a version
has none. Releases before 2.0.9 are described on their GitHub release pages.

## 2.0.14

- A password or key in an Azure endpoint or OpenAI-compatible base URL no
  longer reaches the log (`~/.cortex/cortex.log`, or your `log_file`), or
  the terminal with `debug_mode` on. The HTTP library, and some versions
  of the provider SDKs, log requests with their whole URL; every log line
  now has the credentials of the URLs in it masked. 2.0.13 masked such a
  URL in Cortex's own messages only. If your URL carried a password or a
  key, an earlier log and any CI output captured with `debug_mode` may
  hold it: delete the log and replace that password or key.
- The log, the conversation history and conversations saved with `/save`
  are created as files only you can read, not only as files in the private
  `~/.cortex` folder: a copy or backup of the folder, or a `log_file` set
  elsewhere, no longer leaves a new one readable by other users (a path
  that is a link is left as you set it). The log and history an earlier
  version wrote in `~/.cortex` are tightened when Cortex next opens them
  (one with a second hard link is not); conversations saved earlier keep
  their mode (in the default folder,
  `chmod 600 ~/.cortex/conversations/*.json` tightens them).
- A signal that ends the Cortex process behind the app (SIGTERM, SIGHUP,
  SIGINT) ends it at once whichever of its threads the system delivers it
  to. One delivered to a thread other than the main one was not acted on
  until Cortex's input closed: a process whose input stayed open lived on.
- A 403 or 404 error whose message from the server already ends with a
  full stop no longer reads "... not found.. Check the model name".
- `cortex -p "/update"` says how to update without the app: run the
  installer again.
- An error stays on screen while you run `/status`, `/help` and the other
  commands, so what it says to do next can still be read. It goes when you
  send the next message, choose a model, save a key, or run `/clear`.
- Ctrl+O shows the folded steps and long diffs above a permission prompt
  while it waits. The key did nothing there, though the screen named it.

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
