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


def _print_event(*, session_id: str, event_type: str, payload: dict) -> None:
    if event_type == "message.part.updated":
        part = payload.get("part") or {}
        if part.get("type") == "text":
            sys.stdout.write(str(part.get("delta", "")))
            sys.stdout.flush()
        elif part.get("type") == "tool":
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
        print(f"error: {payload.get('error', '')}", file=sys.stderr)


def run_headless(
    *,
    prompt: str,
    components,
    model: Optional[str] = None,
    full_auto: bool = False,
) -> int:
    from cortex.app.worker_runtime import WorkerRuntime

    config, gpu_validator, conversation_manager = components

    runtime = WorkerRuntime(
        config=config,
        gpu_validator=gpu_validator,
        conversation_manager=conversation_manager,
        rpc_stdin=io.StringIO(),
        rpc_stdout=io.StringIO(),
    )

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

    if model:
        selection = runtime.command_service.execute(
            session_id=session_id, command=f"/model {model}"
        )
        if not bool(selection.get("ok")):
            print(f"model selection failed: {selection.get('message', model)}", file=sys.stderr)
            return 2

    runtime.session_service.create_or_resume(session_id=session_id)
    result = runtime.session_service.submit_user_input(
        session_id=session_id,
        user_input=prompt,
        active_target_input=None,
        stop_sequences=None,
        emit_event=_print_event,
    )

    if not sys.stdout.closed:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return 0 if bool(result.get("ok", True)) else 1
