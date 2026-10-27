"""Headless single-prompt execution (`cortex -p "..."`).

Runs one agent turn through the same WorkerRuntime wiring the TUI uses and
streams the assistant text to stdout. Tool activity goes to stderr so stdout
stays pipeable. Exit codes: 0 success, 1 turn error, 2 setup error.
"""

from __future__ import annotations

import io
import json
import re
import sys
from typing import Optional

from cortex.cloud.types import CloudProvider
from cortex.tooling.orchestrator import skill_command_name
from cortex.tooling.types import PermissionAction, PermissionRule

# Slash commands of the interactive app; a skill's /<name> still runs headless.
_APP_COMMANDS = frozenset(
    {"/help", "/status", "/model", "/login", "/update", "/clear", "/save", "/quit", "/exit"}
)
# What the model is told when a call is refused, so it can say what to do.
_REFUSED = (
    "Not allowed: this headless run (cortex -p) allows edits and shell commands only "
    "with --full-auto."
)
_EXIT_CODE = re.compile(r"exit_code=-?\d+")  # a command that ran and exited non-zero
_GLOB_ESCAPE = re.compile(r"\[([*?\[])\]")  # how "Allow always" escapes a wildcard
# The system prompt's note on a headless run: its reply is the output.
_RUN_NOTE = (
    "This is a single non-interactive run (cortex -p): your final reply is printed as its "
    "result, often read by a script. Reply with the answer itself: do not narrate the steps "
    "you are about to take, and do not offer follow-ups, since nobody can answer."
)
_NO_FULL_AUTO_NOTE = (
    " Edits and shell commands are refused in this run (it has no --full-auto): use the "
    "read, list and search tools, and say what you could not do."
)
_MODEL_STATE_KEYS = ("last_used_cloud_provider", "last_used_cloud_model", "selected_models")


def _allowed_commands(manager) -> str:
    """The commands saved rules let this run use, for the refusal the model
    reads: told only that commands need --full-auto, it gives up on them."""
    commands = [
        _GLOB_ESCAPE.sub(r"\1", rule.pattern)
        for rule in manager.persisted_rules
        if rule.permission == "bash"
        and rule.action == PermissionAction.ALLOW
        and rule.repo in ("", manager.repo)
    ]
    if not commands:
        return ""
    return " These commands are allowed, written exactly so: " + ", ".join(
        f"`{command}`" for command in commands
    ) + "."


def _call_summary(tool: str, arguments) -> str:
    """A tool call as the log shows it: a command as typed, other calls by
    their short arguments (an edit's text is in the diff printed after it)."""
    if not isinstance(arguments, dict):
        return str(arguments)
    command = arguments.get("command")
    if tool == "bash" and isinstance(command, str):
        workdir = arguments.get("workdir")
        where = f" (in {workdir})" if isinstance(workdir, str) and workdir else ""
        return "$ " + command.strip().replace("\n", "\n    ") + where
    shown = []
    for key, value in arguments.items():
        text = value if isinstance(value, str) else json.dumps(value)
        if "\n" in text or len(text) > 80:
            lines = text.count("\n") + 1
            text = f"<{lines} lines>" if lines > 1 else f"<{len(text)} characters>"
        shown.append(f"{key}={text}")
    return " ".join(shown)


def _diff_text(diff: dict) -> str:
    """An edit's diff (the tool result's metadata.diff) as a unified diff."""
    path = str(diff.get("path", ""))
    lines = [f"--- {path}", f"+++ {path}"]
    marks = {"ctx": " ", "del": "-", "add": "+"}
    for hunk in diff.get("hunks", []):
        rows = hunk.get("rows", [])
        old = sum(1 for row in rows if row.get("kind") != "add")
        new = sum(1 for row in rows if row.get("kind") != "del")
        # A side with no lines is numbered from the line before it, as diff does.
        old_start = int(hunk.get("oldStart", 1)) - (0 if old else 1)
        new_start = int(hunk.get("newStart", 1)) - (0 if new else 1)
        lines.append(f"@@ -{old_start},{old} +{new_start},{new} @@")
        lines.extend(marks.get(str(row.get("kind")), " ") + str(row.get("text", "")) for row in rows)
    if diff.get("truncated"):
        lines.append(f"... {diff.get('truncatedRows', 0)} more lines not shown")
    return "\n".join(lines)


def _refusal(selection: dict) -> str:
    """Why a model cannot be used, as a headless run says it: a missing key or
    URL is saved with `cortex login` here, where /login is not at hand."""
    missing = selection.get("missing_key")
    if isinstance(missing, dict):
        login = f"cortex login {missing['provider']}"
        if missing.get("url"):
            login = f"{login} {missing['url']}"
        return (
            f"{missing['provider']} is not authenticated. Save a key with `{login}`, "
            f"or set {missing['variables']}."
        )
    url = selection.get("missing_url")
    if isinstance(url, dict):
        return (
            f"{url['provider']} also needs its {url['what']}: set {url['variable']}, or save it "
            f"with `cortex login {url['provider']} {url['url']}`, which saves a key with it."
        )
    return str(selection.get("message", ""))


def _rules_elsewhere(manager, commands: list[str]) -> list[str]:
    """For each refused command a saved rule allows when Cortex starts in
    another directory, a note saying where: a rule applies only there."""
    notes = []
    for command in dict.fromkeys(commands):
        for rule in manager.persisted_rules:
            if (
                rule.permission == "bash"
                and rule.action == PermissionAction.ALLOW
                and rule.repo not in ("", manager.repo)
                and _GLOB_ESCAPE.sub(r"\1", rule.pattern) == command
            ):
                notes.append(
                    f"note: a saved rule allows `{command}` when Cortex starts in {rule.repo}; "
                    f"this run started in {manager.repo}"
                )
    return notes


def _example_model(runtime) -> str:
    """A --model to suggest: one of a provider whose key is set, else OpenAI's."""
    router = runtime.model_service.cloud_router
    for provider in CloudProvider:
        if router.get_auth_status(provider)[0]:
            names = {CloudProvider.AZURE: "<deployment>", CloudProvider.OPENAI_COMPATIBLE: "<model>"}
            catalog = [
                ref for ref in runtime.model_service.cloud_catalog.list_models()
                if ref.provider == provider
            ]
            model_id = catalog[0].model_id if catalog else names.get(provider, "<model>")
            return f"--model {provider.value}:{model_id}"
    return "--model openai:gpt-5.5"


def _saved_model(config) -> str:
    """The provider:model selected last, from state.yaml ("" if none)."""
    get = getattr(config, "get_state_value", None)
    if not callable(get):
        return ""
    provider = str(get("last_used_cloud_provider", "") or "").strip()
    model_id = str(get("last_used_cloud_model", "") or "").strip()
    return f"{provider}:{model_id}" if provider and model_id else ""


class _Printer:
    """Assistant text to stdout, tool activity to stderr. In a terminal the
    text streams: text that follows a tool call starts a new paragraph, and a
    tool line never starts partway through a line of text. Sent to a pipe or
    a file, stdout gets the answer alone, the text after the last tool call;
    what the model wrote before its tool calls goes to stderr."""

    def __init__(self) -> None:
        self.wrote_text = False
        self.line_open = False
        self.after_tool = False
        self.denied = 0  # tool calls the headless permission policy refused
        self.refused_commands: list[str] = []  # the bash commands among them
        self.inputs: dict[str, dict] = {}  # each call's arguments, by call id
        self.answer_only = not sys.stdout.isatty()
        self.pending: list[str] = []  # answer_only: text since the last tool call

    def _write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self.line_open = not text.endswith("\n")

    def end_line(self) -> None:
        if self.line_open and not sys.stdout.closed:
            self._write("\n")

    def finish(self) -> None:
        """Print the answer held back for a pipe, and end the last line."""
        if self.pending:
            self._write("".join(self.pending).strip("\n"))
            self.pending = []
        self.end_line()

    def event(self, *, session_id: str, event_type: str, payload: dict) -> None:
        if event_type == "message.part.updated":
            part = payload.get("part") or {}
            if part.get("type") == "text":
                delta = str(part.get("delta", ""))
                if not delta:
                    return
                if self.answer_only:
                    self.pending.append(delta)
                    self.wrote_text = True
                    return
                if self.after_tool and self.wrote_text:
                    self.end_line()
                    self._write("\n")
                self.after_tool = False
                self.wrote_text = True
                self._write(delta)
            elif part.get("type") == "tool":
                if self.pending:
                    # Words before a tool call are not the answer.
                    print("".join(self.pending).strip(), file=sys.stderr)
                    self.pending = []
                self.end_line()
                self.after_tool = True
                state = str(part.get("state", ""))
                tool = str(part.get("tool", ""))
                if state == "running":
                    arguments = part.get("input") or {}
                    self.inputs[str(part.get("call_id", ""))] = arguments
                    summary = _call_summary(tool, arguments)
                    print(f"[tool] {tool} {summary}".rstrip(), file=sys.stderr)
                elif state == "error" or part.get("ok") is False:
                    error = str(part.get("error", ""))
                    if error.startswith(_REFUSED):
                        self.denied += 1
                        command = self.inputs.get(str(part.get("call_id", "")), {}).get("command")
                        if tool == "bash" and isinstance(command, str):
                            self.refused_commands.append(command)
                        print(f"[tool] {tool} refused (needs --full-auto)", file=sys.stderr)
                    elif _EXIT_CODE.fullmatch(error):
                        code = error.split("=", 1)[1]
                        print(f"[tool] {tool} exited with code {code}", file=sys.stderr)
                    else:
                        print(f"[tool] {tool} failed: {error}", file=sys.stderr)
                elif state == "completed":
                    output = str(part.get("output", "") or "")
                    print(f"[tool] {tool} ok ({len(output)} chars)", file=sys.stderr)
                    diff = (part.get("metadata") or {}).get("diff")
                    if isinstance(diff, dict):
                        print(_diff_text(diff), file=sys.stderr)
            return
        if event_type == "session.error":
            self.end_line()
            print(f"error: {payload.get('error', '')}", file=sys.stderr)


def run_headless(
    *,
    prompt: str,
    components,
    model: Optional[str] = None,
    full_auto: bool = False,
) -> int:
    from cortex.app.worker_runtime import WorkerRuntime

    config, conversation_manager = components
    # There is no session notice in a headless run: settings that were not
    # valid are reported on stderr.
    for warning in getattr(config, "warnings", []):
        print(f"warning: {warning}", file=sys.stderr)

    runtime = WorkerRuntime(
        config=config,
        conversation_manager=conversation_manager,
        rpc_stdin=io.StringIO(),
        rpc_stdout=io.StringIO(),
    )

    # Nobody can answer a permission prompt here (a saved rule may still ask).
    runtime.tooling_bridge.interactive = False

    # Headless permission policy: reads stay allowed via default rules; writes
    # and bash are auto-approved with --full-auto and denied otherwise (there
    # is no interactive prompt to ask). Persisted user rules still win.
    manager = runtime.tooling_orchestrator.permission_manager
    if full_auto:
        manager.base_rules.insert(
            0, PermissionRule(permission="*", pattern="*", action=PermissionAction.ALLOW)
        )
    else:
        manager.base_rules.insert(
            0,
            PermissionRule(
                permission="*",
                pattern="*",
                action=PermissionAction.DENY,
                reason=_REFUSED + _allowed_commands(manager),
            ),
        )
        manager.base_rules.extend(
            PermissionRule(permission=name, pattern="*", action=PermissionAction.ALLOW)
            for name in ("read", "list", "grep")
        )

    note = _RUN_NOTE
    if not full_auto:
        note += _NO_FULL_AUTO_NOTE + _allowed_commands(manager)
    runtime.tooling_orchestrator.run_note = note

    session_id = "headless"

    command = prompt.strip().split(maxsplit=1)[0].lower() if prompt.strip() else ""
    if command in _APP_COMMANDS:
        hints = {
            "/model": " Pick the model with --model provider:model.",
            "/login": " Here, save a key with `cortex login PROVIDER [URL]`, which reads it from "
            "stdin.",
        }
        hint = hints.get(command, "")
        print(
            f"error: {command} is a command of the interactive app: run `cortex` and type it "
            f"there.{hint}",
            file=sys.stderr,
        )
        return 2

    # The state a --model run would overwrite, put back if the provider
    # refuses the model: a mistyped name must not become the saved choice.
    saved_state = {key: config.get_state_value(key) for key in _MODEL_STATE_KEYS}
    if model:
        selection = runtime.command_service.execute(
            session_id=session_id, command=f"/model {model}"
        )
        if not bool(selection.get("ok")):
            message = _refusal(selection) or model
            if message.startswith("Usage: /model"):
                message = f"pass --model provider:model, for example {_example_model(runtime)}"
            print(f"error: {message}", file=sys.stderr)
            return 2

    elif getattr(runtime.model_service.active_target, "cloud_model", None) is None:
        saved = _saved_model(config)
        if saved:
            # Remembered, but not usable now (most often: its key is not set here).
            provider, _, model_id = saved.partition(":")
            reason = runtime.model_service.select_cloud_model(provider=provider, model_id=model_id)
            message = _refusal(reason)
            print(f"error: {saved} cannot be used: {message}", file=sys.stderr)
            return 2
        print(
            "error: no model selected. Pass one with --model provider:model, "
            f"for example {_example_model(runtime)} (see cortex --help).",
            file=sys.stderr,
        )
        return 2

    name = skill_command_name(prompt.strip())
    skills = runtime.tooling_orchestrator.available_skills()
    if name and name not in skills:
        # Sent as written, as any prompt is; the run says it ran no skill.
        sys.stderr.write(
            f"note: no skill is named {name} (skills here: {', '.join(sorted(skills)) or 'none'}); "
            "the prompt is sent as written\n"
        )

    printer = _Printer()
    runtime.session_service.create_or_resume(session_id=session_id)
    result = runtime.session_service.submit_user_input(
        session_id=session_id,
        user_input=prompt,
        active_target_input=None,
        stop_sequences=None,
        emit_event=printer.event,
    )
    printer.finish()
    ok = bool(result.get("ok", True))
    if model and not ok and not (printer.wrote_text or printer.after_tool):
        for key, value in saved_state.items():
            config.set_state_value(key, value)
    if printer.denied and not full_auto:
        print(
            f"note: {printer.denied} edit or command call(s) were not allowed; "
            "pass --full-auto to allow edits and shell commands",
            file=sys.stderr,
        )
        for note in _rules_elsewhere(manager, printer.refused_commands):
            print(note, file=sys.stderr)
    # Last, as the docs say: the model the answer came from.
    served = runtime.model_service.last_turn_provenance
    if served:
        check = "verified" if served.get("verified") else "unverified"
        print(f"answered by {served.get('label')} ({check})", file=sys.stderr)

    return 0 if ok else 1
