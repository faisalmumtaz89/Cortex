"""Headless single-prompt execution (`cortex -p "..."`).

Runs one agent turn through the same WorkerRuntime wiring the TUI uses and
streams the assistant text to stdout. Tool activity goes to stderr so stdout
stays pipeable. Exit codes: 0 success, 1 turn error, 2 setup error.
"""

from __future__ import annotations

import io
import sys
from typing import Optional

from cortex.tooling.types import PermissionAction, PermissionRule

# Slash commands of the interactive app; a skill's /<name> still runs headless.
_APP_COMMANDS = frozenset(
    {"/help", "/status", "/model", "/login", "/update", "/clear", "/save", "/quit", "/exit"}
)


class _Printer:
    """Assistant text to stdout, tool activity to stderr. Text that follows a
    tool call starts a new paragraph, and a tool line never starts partway
    through a line of text when both streams share a terminal."""

    def __init__(self) -> None:
        self.wrote_text = False
        self.line_open = False
        self.after_tool = False

    def _write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self.line_open = not text.endswith("\n")

    def end_line(self) -> None:
        if self.line_open and not sys.stdout.closed:
            self._write("\n")

    def event(self, *, session_id: str, event_type: str, payload: dict) -> None:
        if event_type == "message.part.updated":
            part = payload.get("part") or {}
            if part.get("type") == "text":
                delta = str(part.get("delta", ""))
                if not delta:
                    return
                if self.after_tool and self.wrote_text:
                    self.end_line()
                    self._write("\n")
                self.after_tool = False
                self.wrote_text = True
                self._write(delta)
            elif part.get("type") == "tool":
                self.end_line()
                self.after_tool = True
                state = str(part.get("state", ""))
                tool = str(part.get("tool", ""))
                if state == "running":
                    print(f"[tool] {tool} {part.get('input', {})}", file=sys.stderr)
                elif state == "error" or part.get("ok") is False:
                    print(f"[tool] {tool} failed: {part.get('error', '')}", file=sys.stderr)
                elif state == "completed":
                    output = str(part.get("output", "") or "")
                    print(f"[tool] {tool} ok ({len(output)} chars)", file=sys.stderr)
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
            0, PermissionRule(permission="*", pattern="*", action=PermissionAction.DENY)
        )
        manager.base_rules.extend(
            PermissionRule(permission=name, pattern="*", action=PermissionAction.ALLOW)
            for name in ("read", "list", "grep")
        )

    session_id = "headless"

    command = prompt.strip().split(maxsplit=1)[0].lower() if prompt.strip() else ""
    if command in _APP_COMMANDS:
        hint = " Pick the model with --model provider:model." if command == "/model" else ""
        print(
            f"error: {command} is a command of the interactive app: run `cortex` and type it "
            f"there.{hint}",
            file=sys.stderr,
        )
        return 2

    if model:
        selection = runtime.command_service.execute(
            session_id=session_id, command=f"/model {model}"
        )
        if not bool(selection.get("ok")):
            print(f"model selection failed: {selection.get('message', model)}", file=sys.stderr)
            return 2

    elif getattr(runtime.model_service.active_target, "cloud_model", None) is None:
        print(
            "error: no model selected. Pass one with --model provider:model, "
            "for example --model openai:gpt-5.5 (see cortex --help).",
            file=sys.stderr,
        )
        return 2

    printer = _Printer()
    runtime.session_service.create_or_resume(session_id=session_id)
    result = runtime.session_service.submit_user_input(
        session_id=session_id,
        user_input=prompt,
        active_target_input=None,
        stop_sequences=None,
        emit_event=printer.event,
    )
    printer.end_line()

    return 0 if bool(result.get("ok", True)) else 1
