"""Tool-aware generation orchestrator."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, cast

from cortex.cloud.types import CloudModelRef, CloudProvider
from cortex.tooling.agent_prompt import build_system_prompt
from cortex.tooling.builtin.skill import skill_text
from cortex.tooling.permissions import (
    PermissionDecision,
    PermissionDeniedError,
    PermissionManager,
    PermissionRequest,
    default_rules,
)
from cortex.tooling.provenance import verify_turn_provenance
from cortex.tooling.registry import ToolRegistry
from cortex.tooling.skills import Skill, load_skills
from cortex.tooling.stream_normalizer import merge_stream_text
from cortex.tooling.types import (
    AssistantTurnResult,
    ErrorEvent,
    FinishEvent,
    ModelEvent,
    ReasoningEvent,
    ReplyCutOffError,
    TextDeltaEvent,
    ToolCall,
    ToolCallEvent,
    ToolExecutionState,
    ToolResult,
    ToolResultEvent,
    TurnInterrupt,
    TurnInterruptedError,
)

logger = logging.getLogger(__name__)

NO_TOOLS_SYSTEM_INSTRUCTION = (
    "Tool usage is disabled for this session. "
    "Do not emit <tool_calls> blocks or tool JSON. "
    "Do not claim generic platform limitations. "
    "If user asks to inspect files/code, clearly say tooling is disabled and ask to enable read-only tools."
)


# Tool calls and results from the latest turns reach the model whole; in older
# turns each result and each long argument keeps only its first and last
# OLD_TEXT_CHARS // 2 characters.
FULL_TURNS = 3
OLD_TEXT_CHARS = 1500


def conversation_history(conversation) -> List[Dict[str, object]]:
    """The conversation as Chat Completions messages, with each assistant
    reply's tool calls and their results."""
    if conversation is None:
        return []
    messages = [m for m in conversation.messages if m.role.value in {"system", "user", "assistant"}]
    user_turns = [index for index, m in enumerate(messages) if m.role.value == "user"]
    full_from = user_turns[-FULL_TURNS] if len(user_turns) >= FULL_TURNS else 0
    call_ids: Set[str] = set()
    history: List[Dict[str, object]] = []
    for index, message in enumerate(messages):
        if message.role.value == "assistant" and message.parts:
            history.extend(_assistant_history(message.parts, index >= full_from, call_ids))
        elif (message.content or "").strip():
            history.append({"role": message.role.value, "content": message.content.strip()})
    return history


def _assistant_history(parts, full: bool, call_ids: Set[str]) -> List[Dict[str, object]]:
    """One assistant reply as messages: each step's text, reasoning and tool
    calls, each call followed by its result. Calls that never got a result are
    left out."""
    history: List[Dict[str, object]] = []
    text = ""
    reasoning: Dict[str, object] = {}
    calls: List[Dict[str, object]] = []
    outputs: List[Dict[str, object]] = []
    waiting: Dict[str, Dict[str, Any]] = {}

    def close_step() -> None:
        nonlocal text, reasoning, calls, outputs
        if calls:
            history.append(
                {"role": "assistant", "content": text or None, **reasoning, "tool_calls": calls}
            )
            history.extend(outputs)
        elif text.strip():
            history.append({"role": "assistant", "content": text, **reasoning})
        text, reasoning, calls, outputs = "", {}, [], []

    for part in parts:
        if part.get("type") in ("text", "reasoning") and calls:
            close_step()
        if part.get("type") == "text":
            text += part.get("delta") or ""
        elif part.get("type") == "reasoning" and part.get("field"):
            reasoning = {str(part["field"]): str(part.get("text") or "")}
        elif part.get("type") == "tool" and part.get("state") == "running":
            waiting[str(part.get("call_id"))] = part
        elif part.get("type") == "tool":
            call = waiting.pop(str(part.get("call_id")), None)
            if call is None:
                continue
            call_id = _unique_call_id(str(part.get("call_id")), call_ids)
            arguments = call.get("input") or {}
            if not full:
                arguments = {key: _shorten(value) for key, value in arguments.items()}
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": _API_NAME.sub("_", str(call.get("tool") or "")) or "unknown_tool",
                        "arguments": json.dumps(arguments),
                    },
                }
            )
            output = str(part.get("output") or "")
            if not part.get("ok"):
                output = f"error: {part.get('error') or ''}\n{output}".strip()
            outputs.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output if full else _shorten(output),
                }
            )
    close_step()
    return history


# Tool names and call ids in the form every provider API accepts.
_API_NAME = re.compile(r"[^a-zA-Z0-9_-]")


_SKILL_COMMAND = re.compile(r"/([a-z0-9]+(?:-[a-z0-9]+)*)(?=\s|$)")


def expand_skill_commands(
    messages: List[Dict[str, object]], skills: Dict[str, Skill]
) -> Optional[Skill]:
    """Give each user message that starts with /<skill name> the skill's
    instructions, as if the model had loaded it. Returns the skill the last
    user message runs, if any."""
    invoked: Optional[Skill] = None
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        match = _SKILL_COMMAND.match(content) if isinstance(content, str) else None
        skill = skills.get(match.group(1)) if match else None
        invoked = skill
        if skill is None:
            continue
        messages[index] = {
            **message,
            "content": f"{content}\n\nThe user ran the {skill.name} skill for this "
            f"request. Its instructions:\n\n{skill_text(skill)}",
        }
    return invoked


class _ToolLimit:
    """The tools the skills loaded in a turn allow: those every one of them allows."""

    def __init__(self) -> None:
        self.skills: List[str] = []
        self.tools: Optional[Set[str]] = None

    def narrow(self, skill: Skill) -> None:
        if skill.allowed_tools is None:
            return
        if skill.name not in self.skills:
            self.skills.append(skill.name)
        allowed = set(skill.allowed_tools)
        self.tools = allowed if self.tools is None else self.tools & allowed

    def refusal(self, tool: str) -> Optional[str]:
        if self.tools is None or tool == "skill" or tool in self.tools:
            return None
        allowed = ", ".join(sorted(self.tools)) or "none"
        if len(self.skills) == 1:
            return (
                f"The {self.skills[0]} skill does not use {tool}; for the rest of this turn "
                f"only these tools run: {allowed}. Do not work around this with other tools: "
                "finish within them, or say what is needed and stop."
            )
        return (
            f"The {' and '.join(self.skills)} skills together allow only these tools: "
            f"{allowed}; {tool} is not one of them for the rest of this turn. Do not work "
            "around this with other tools: finish within them, or say what is needed and stop."
        )


def _unique_call_id(call_id: str, taken: Set[str]) -> str:
    """The id made unique in the conversation, within the 64 characters the
    OpenAI Responses API allows."""
    base = _API_NAME.sub("_", call_id)[:64] or "call"
    unique, count = base, 1
    while unique in taken:
        count += 1
        suffix = f"_{count}"
        unique = base[: 64 - len(suffix)] + suffix
    taken.add(unique)
    return unique


def _shorten(value: Any) -> Any:
    if not isinstance(value, str) or len(value) <= OLD_TEXT_CHARS:
        return value
    half = OLD_TEXT_CHARS // 2
    omitted = len(value) - 2 * half
    return f"{value[:half]}\n[... {omitted} characters omitted ...]\n{value[-half:]}"


class ToolingOrchestrator:
    """Coordinates generation, tool execution, and permission checks."""

    def __init__(self, *, cli):
        self.cli = cli
        self.permission_manager = PermissionManager(rules=default_rules())
        # What the system prompt says about the run itself (headless mode).
        self.run_note: Optional[str] = None

    def _tooling_flags(self) -> dict:
        tools_cfg = getattr(self.cli.config, "tools", None)
        tools_enabled = bool(getattr(tools_cfg, "tools_enabled", False))
        tools_profile = str(getattr(tools_cfg, "tools_profile", "off") or "off")
        max_iterations = int(getattr(tools_cfg, "tools_max_iterations", 100) or 100)
        return {
            "enabled": tools_enabled and tools_profile != "off",
            "profile": tools_profile,
            "max_iterations": max(1, max_iterations),
        }

    def available_skills(self) -> Dict[str, Skill]:
        """The skills a turn would offer: none when tools are off."""
        if not self._tooling_flags()["enabled"]:
            return {}
        return load_skills(Path.cwd())

    def _extract_permission_patterns(self, call: ToolCall, repo_root: Path) -> List[str]:
        """What the call does, as permission rules match it: for bash the command
        exactly as it will run, with the directory it runs in; for other tools
        the file they touch, relative to the repository. A call without either
        gets "*", which "Allow always" never saves."""
        args = call.arguments or {}

        def inside(raw: str) -> Optional[str]:
            """`raw` resolved as the tools resolve it, relative to the
            repository; None when it is outside it or not a usable path."""
            try:
                path = Path(raw)
                path = (path if path.is_absolute() else repo_root / path).resolve()
            except (TypeError, ValueError, OSError):
                return None
            if not path.is_relative_to(repo_root):
                return None
            return path.relative_to(repo_root).as_posix()

        if call.name == "bash":
            command, workdir = args.get("command"), args.get("workdir")
            if not isinstance(command, str) or not command.strip():
                return ["*"]
            if not workdir:
                return [command]
            # Resolved as the bash tool resolves it, and written as a script that
            # runs the command there or not at all, as the tool does: "./"
            # keeps `cd` from reading the directory as an option or looking it
            # up in CDPATH, and "|| exit 1" stops if it is missing.
            where = inside(os.path.expanduser(workdir)) if isinstance(workdir, str) else None
            if where is None:
                return ["*"]
            if where == ".":
                return [command]
            return [f"cd {shlex.quote('./' + where)} || exit 1\n{command}"]
        # Edit tools strip the path before they resolve it; so does the pattern.
        path = args.get("path")
        where = inside(path.strip()) if isinstance(path, str) and path.strip() else None
        return [where] if where is not None else ["*"]

    def _prompt_permission(self, request: PermissionRequest) -> PermissionDecision:
        prompt_fn = getattr(self.cli, "prompt_tool_permission", None)
        if callable(prompt_fn):
            return cast(PermissionDecision, prompt_fn(request))
        return PermissionDecision.REJECT

    def _execute_tool_call(
        self,
        *,
        registry: ToolRegistry,
        session_id: str,
        call: ToolCall,
        interrupt: Optional[TurnInterrupt] = None,
    ) -> ToolResult:
        if not registry.has(call.name):
            return ToolResult(
                id=call.id,
                name=call.name,
                state=ToolExecutionState.ERROR,
                ok=False,
                error=f"tool not available under current profile: {call.name}",
            )

        permission = registry.permission_for(call.name)
        patterns = self._extract_permission_patterns(call, registry.repo_root)

        try:
            self.permission_manager.request(
                permission=permission,
                patterns=patterns,
                metadata={"tool": call.name, "arguments": call.arguments},
                prompt_callback=self._prompt_permission,
                preview=lambda: registry.preview(call=call, session_id=session_id),
            )
        except PermissionDeniedError as exc:
            return ToolResult(
                id=call.id,
                name=call.name,
                state=ToolExecutionState.ERROR,
                ok=False,
                error=str(exc),
            )

        return registry.execute(call=call, session_id=session_id, interrupt=interrupt)

    def _record_event(
        self,
        *,
        event: ModelEvent,
        result: AssistantTurnResult,
        on_event: Optional[Callable[[ModelEvent], None]],
        first_text_seen: Dict[str, float],
        started_at: float,
    ) -> None:
        if isinstance(event, TextDeltaEvent):
            delta, assembled = merge_stream_text(event.delta, result.text)
            if not delta:
                return
            normalized_event = TextDeltaEvent(delta=delta)
            if on_event is not None:
                on_event(normalized_event)

            result.text = assembled
            result.token_count += 1
            if "first" not in first_text_seen:
                first_text_seen["first"] = time.time() - started_at
            result.parts.append({"type": "text", "delta": delta})
            return

        if isinstance(event, ToolCallEvent):
            result.parts.append(
                {
                    "type": "tool",
                    "state": "running",
                    "call_id": event.call.id,
                    "tool": event.call.name,
                    "input": event.call.arguments,
                }
            )

        if isinstance(event, ToolResultEvent):
            result.parts.append(
                {
                    "type": "tool",
                    "state": event.result.state.value,
                    "call_id": event.result.id,
                    "tool": event.result.name,
                    "ok": event.result.ok,
                    "output": event.result.output,
                    "error": event.result.error,
                    "metadata": event.result.metadata,
                }
            )

        if isinstance(event, ReasoningEvent):
            result.parts.append({"type": "reasoning", "field": event.field, "text": event.text})

        if isinstance(event, ErrorEvent):
            result.parts.append({"type": "error", "error": event.error, "metadata": event.metadata})

        if isinstance(event, FinishEvent):
            result.parts.append({"type": "finish", "reason": event.reason})
            if event.provenance is not None:
                result.provenance = dict(event.provenance)

        # Recorded first: a tool result stays in the turn's parts even when the
        # turn is interrupted while it is reported. A call recorded this way
        # without a result is left out of the history.
        if on_event is not None:
            on_event(event)

    def run_turn(
        self,
        user_input,
        active_target,
        conversation,
        *,
        stop_sequences: Optional[List[str]] = None,
        on_event: Optional[Callable[[ModelEvent], None]] = None,
        on_retry: Optional[Callable[[int, int, str], None]] = None,
        interrupt: Optional[TurnInterrupt] = None,
    ) -> AssistantTurnResult:
        """Run one generation turn and return structured output."""
        flags = self._tooling_flags()
        tools_enabled = flags["enabled"]
        max_iterations = flags["max_iterations"]

        profile = flags["profile"] if tools_enabled else "off"
        registry = ToolRegistry(repo_root=Path.cwd(), profile=profile)
        session_id = conversation.conversation_id if conversation else "session"

        result = AssistantTurnResult(text="")
        started_at = time.time()
        first_text_seen: Dict[str, float] = {}

        model_ref = active_target.cloud_model
        if model_ref is None:
            raise RuntimeError("No model loaded. Pick one with /model.")

        messages = conversation_history(conversation)
        if not messages:
            messages = [{"role": "user", "content": user_input}]

        invoked = expand_skill_commands(messages, registry.skills) if tools_enabled else None
        if tools_enabled:
            messages = [
                {
                    "role": "system",
                    "content": build_system_prompt(
                        cwd=registry.repo_root, skills=registry.skills, run_note=self.run_note
                    ),
                },
                *messages,
            ]
        else:
            messages = [{"role": "system", "content": NO_TOOLS_SYSTEM_INSTRUCTION}, *messages]

        tool_specs = registry.specs() if tools_enabled else []

        # The tools loaded skills limit the rest of this turn to: each skill with
        # allowed-tools narrows the limit, and none lifts it. The skill tool always runs.
        limit = _ToolLimit()
        if invoked is not None:
            limit.narrow(invoked)

        def execute_tool(call: ToolCall) -> ToolResult:
            refusal = limit.refusal(call.name)
            if refusal:
                return ToolResult(
                    id=call.id,
                    name=call.name,
                    state=ToolExecutionState.ERROR,
                    ok=False,
                    error=refusal,
                )
            tool_result = self._execute_tool_call(
                registry=registry,
                session_id=session_id,
                call=call,
                interrupt=interrupt,
            )
            metadata = tool_result.metadata or {}
            if call.name == "skill" and tool_result.ok and metadata.get("file") is None:
                loaded = registry.skills.get(str(metadata.get("skill")))
                if loaded is not None:
                    limit.narrow(loaded)
            return tool_result

        events = self.cli.cloud_router.stream_events(
            model_ref=model_ref,
            messages=messages,
            max_tokens=self.cli.config.inference.max_tokens,
            temperature=self.cli.config.inference.temperature,
            top_p=self.cli.config.inference.top_p,
            tools=tool_specs,
            tool_choice="auto",
            tool_executor=execute_tool if tools_enabled else None,
            max_tool_iterations=max_iterations,
            on_retry=on_retry,
            interrupt=interrupt,
        )

        finish_reason = ""
        try:
            for event in events:
                if isinstance(event, FinishEvent):
                    finish_reason = event.reason
                self._record_event(
                    event=event,
                    result=result,
                    on_event=on_event,
                    first_text_seen=first_text_seen,
                    started_at=started_at,
                )
        except TurnInterruptedError as exc:
            exc.parts = result.parts
            raise

        self._verify_turn_provenance(
            result=result,
            model_ref=model_ref,
            on_event=on_event,
        )
        if finish_reason == "length":
            raise ReplyCutOffError(
                "The model reached the output limit "
                f"(max_tokens={self.cli.config.inference.max_tokens}) before finishing its "
                "reply. Raise max_tokens in ~/.cortex/config.yaml (or CORTEX_MAX_TOKENS).",
                result.parts,
            )
        if finish_reason == "tool_limit":
            raise ReplyCutOffError(
                f"The turn reached its limit of model requests "
                f"(tools_max_iterations={max_iterations}). In an interactive session its "
                "tool calls stay in the conversation, so a new message lets the model "
                "continue; to allow longer turns, raise tools_max_iterations in "
                "~/.cortex/config.yaml.",
                result.parts,
            )
        if finish_reason == "context_window":
            raise ReplyCutOffError(
                "The conversation filled the model's context window before the reply "
                "finished. Start a new conversation with /clear.",
                result.parts,
            )

        result.elapsed_seconds = time.time() - started_at
        result.first_token_latency_seconds = first_text_seen.get("first")
        return result

    def _verify_turn_provenance(
        self,
        *,
        result: AssistantTurnResult,
        model_ref: CloudModelRef,
        on_event: Optional[Callable[[ModelEvent], None]],
    ) -> None:
        """Fail the turn unless the response proved it came from the
        requested model (see cortex/tooling/provenance.py)."""
        expected_endpoint: Optional[str] = None
        if model_ref.provider == CloudProvider.OPENAI_COMPATIBLE:
            expected_endpoint = self.cli.cloud_router.openai_compatible_base_url()

        verdict = verify_turn_provenance(
            provider=model_ref.provider,
            requested_model=model_ref.model_id,
            provenance=result.provenance,
            expected_endpoint=expected_endpoint,
        )

        if not verdict.ok:
            error = (
                f"Model provenance mismatch: asked {model_ref.selector}, but {verdict.reason} "
                f"— turn rejected."
            )
            if on_event is not None:
                on_event(ErrorEvent(error=error, metadata={"provenance": result.provenance or {}}))
            raise RuntimeError(error)

        result.provenance_verified = True
        served = model_ref.selector
        result.served_model_label = f"{served} (scripted)" if verdict.scripted else served
