"""Tool-aware generation orchestrator."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, cast

from cortex.conversation_manager import MessageRole
from cortex.inference_engine import GenerationRequest
from cortex.tooling.local_protocol import format_tool_results, parse_tool_calls, strip_tool_blocks
from cortex.tooling.permissions import (
    PermissionDecision,
    PermissionDeniedError,
    PermissionManager,
    PermissionRequest,
)
from cortex.tooling.registry import ToolRegistry
from cortex.tooling.stream_normalizer import merge_stream_text
from cortex.tooling.types import (
    AssistantTurnResult,
    ErrorEvent,
    FinishEvent,
    ModelEvent,
    TextDeltaEvent,
    ToolCall,
    ToolCallEvent,
    ToolExecutionState,
    ToolResult,
    ToolResultEvent,
)

logger = logging.getLogger(__name__)

NO_TOOLS_SYSTEM_INSTRUCTION = (
    "Tool usage is disabled for this session. "
    "Do not emit <tool_calls> blocks or tool JSON. "
    "Do not claim generic platform limitations. "
    "If user asks to inspect files/code, clearly say tooling is disabled and ask to enable read-only tools."
)

TOOLS_SYSTEM_INSTRUCTION_TEMPLATE = (
    "You are running inside the user's local repository at: {cwd}. "
    "For requests about codebase/files/functions/scripts, use the available tools to inspect real files before concluding. "
    "Do not say you lack repository access when tools are available. "
    "If a tool is rejected or unavailable, explain that constraint and continue with best effort."
)


class ToolingOrchestrator:
    """Coordinates generation, tool execution, and permission checks."""

    def __init__(self, *, cli):
        self.cli = cli
        self.permission_manager = PermissionManager()

    def _tooling_flags(self) -> dict:
        tools_cfg = getattr(self.cli.config, "tools", None)
        tools_enabled = bool(getattr(tools_cfg, "tools_enabled", False))
        tools_profile = str(getattr(tools_cfg, "tools_profile", "off") or "off")
        tools_local_mode = str(getattr(tools_cfg, "tools_local_mode", "disabled") or "disabled")
        max_iterations = int(getattr(tools_cfg, "tools_max_iterations", 4) or 4)
        return {
            "enabled": tools_enabled and tools_profile != "off",
            "profile": tools_profile,
            "local_mode": tools_local_mode,
            "max_iterations": max(1, max_iterations),
        }

    def _build_message_window(self, *, conversation, window: int = 30) -> List[Dict[str, object]]:
        messages: List[Dict[str, object]] = []
        if conversation is None or not conversation.messages:
            return messages

        for message in conversation.messages[-window:]:
            content = (message.content or "").strip()
            if not content:
                continue
            role = message.role.value
            if role not in {"system", "user", "assistant"}:
                continue
            messages.append({"role": role, "content": content})
        return messages

    def _extract_permission_patterns(self, call: ToolCall) -> List[str]:
        args = call.arguments or {}
        patterns: List[str] = []
        for key in ("path", "filePath", "workdir"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                patterns.append(value.strip())
        if not patterns:
            patterns.append("*")
        return patterns

    def _prompt_permission(self, request: PermissionRequest) -> PermissionDecision:
        prompt_fn = getattr(self.cli, "prompt_tool_permission", None)
        if callable(prompt_fn):
            return cast(PermissionDecision, prompt_fn(request))
        return PermissionDecision.REJECT

    def _execute_tool_call(self, *, registry: ToolRegistry, session_id: str, call: ToolCall) -> ToolResult:
        if not registry.has(call.name):
            return ToolResult(
                id=call.id,
                name=call.name,
                state=ToolExecutionState.ERROR,
                ok=False,
                error=f"tool not available under current profile: {call.name}",
            )

        permission = registry.permission_for(call.name)
        patterns = self._extract_permission_patterns(call)

        try:
            self.permission_manager.request(
                permission=permission,
                patterns=patterns,
                metadata={"tool": call.name, "arguments": call.arguments},
                session_id=session_id,
                prompt_callback=self._prompt_permission,
            )
        except PermissionDeniedError as exc:
            return ToolResult(
                id=call.id,
                name=call.name,
                state=ToolExecutionState.ERROR,
                ok=False,
                error=str(exc),
            )

        return registry.execute(call=call, session_id=session_id)

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

        if on_event is not None:
            on_event(event)

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
            return

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
            return

        if isinstance(event, ErrorEvent):
            result.parts.append({"type": "error", "error": event.error, "metadata": event.metadata})
            return

        if isinstance(event, FinishEvent):
            result.parts.append({"type": "finish", "reason": event.reason})
            return

    def _build_local_request(
        self,
        *,
        user_input: str,
        stop_sequences: Optional[List[str]],
        tools_enabled: bool,
    ) -> GenerationRequest:
        prompt = self.cli._format_prompt_with_chat_template(user_input, include_user=False)
        if not tools_enabled:
            prompt = f"{NO_TOOLS_SYSTEM_INSTRUCTION}\n\n{prompt}"
        return GenerationRequest(
            prompt=prompt,
            max_tokens=self.cli.config.inference.max_tokens,
            temperature=self.cli.config.inference.temperature,
            top_p=self.cli.config.inference.top_p,
            top_k=self.cli.config.inference.top_k,
            repetition_penalty=self.cli.config.inference.repetition_penalty,
            stream=self.cli.config.inference.stream_output,
            seed=self.cli.config.inference.seed if self.cli.config.inference.seed >= 0 else None,
            stop_sequences=stop_sequences or [],
        )

    def _run_local(
        self,
        *,
        user_input: str,
        conversation,
        registry: ToolRegistry,
        tools_enabled: bool,
        local_tools_experimental: bool,
        max_iterations: int,
        stop_sequences: Optional[List[str]],
        on_event: Optional[Callable[[ModelEvent], None]],
        result: AssistantTurnResult,
        first_text_seen: Dict[str, float],
        started_at: float,
    ) -> None:
        session_id = conversation.conversation_id if conversation else "session_local"

        if tools_enabled and local_tools_experimental:
            iterations = 0
            while iterations < max_iterations:
                iterations += 1
                request = self._build_local_request(
                    user_input=user_input,
                    stop_sequences=stop_sequences,
                    tools_enabled=tools_enabled,
                )
                generated = "".join(self.cli.inference_engine.generate(request))

                calls, parse_error = parse_tool_calls(generated)
                if parse_error and not calls:
                    cleaned = strip_tool_blocks(generated)
                    if cleaned:
                        self._record_event(
                            event=TextDeltaEvent(delta=cleaned),
                            result=result,
                            on_event=on_event,
                            first_text_seen=first_text_seen,
                            started_at=started_at,
                        )
                    self._record_event(
                        event=ErrorEvent(error=f"Local tool parse warning: {parse_error}"),
                        result=result,
                        on_event=on_event,
                        first_text_seen=first_text_seen,
                        started_at=started_at,
                    )
                    self._record_event(
                        event=FinishEvent(reason="stop"),
                        result=result,
                        on_event=on_event,
                        first_text_seen=first_text_seen,
                        started_at=started_at,
                    )
                    return

                if calls:
                    if iterations >= max_iterations:
                        raise RuntimeError("Tool loop limit reached for local model response")

                    tool_results_payload = []
                    for call in calls:
                        self._record_event(
                            event=ToolCallEvent(call=call),
                            result=result,
                            on_event=on_event,
                            first_text_seen=first_text_seen,
                            started_at=started_at,
                        )
                        tool_result = self._execute_tool_call(
                            registry=registry,
                            session_id=session_id,
                            call=call,
                        )
                        self._record_event(
                            event=ToolResultEvent(result=tool_result),
                            result=result,
                            on_event=on_event,
                            first_text_seen=first_text_seen,
                            started_at=started_at,
                        )
                        tool_results_payload.append(
                            {
                                "id": tool_result.id,
                                "name": tool_result.name,
                                "ok": tool_result.ok,
                                "result": {
                                    "output": tool_result.output,
                                    "metadata": tool_result.metadata,
                                },
                                "error": tool_result.error,
                            }
                        )

                    conversation.add_message(MessageRole.SYSTEM, format_tool_results(tool_results_payload))
                    continue

                final_text = strip_tool_blocks(generated)
                if final_text:
                    self._record_event(
                        event=TextDeltaEvent(delta=final_text),
                        result=result,
                        on_event=on_event,
                        first_text_seen=first_text_seen,
                        started_at=started_at,
                    )
                self._record_event(
                    event=FinishEvent(reason="stop"),
                    result=result,
                    on_event=on_event,
                    first_text_seen=first_text_seen,
                    started_at=started_at,
                )
                return

            raise RuntimeError("Tool loop limit reached")

        request = self._build_local_request(
            user_input=user_input,
            stop_sequences=stop_sequences,
            tools_enabled=tools_enabled,
        )
        for token in self.cli.inference_engine.generate(request):
            self._record_event(
                event=TextDeltaEvent(delta=str(token)),
                result=result,
                on_event=on_event,
                first_text_seen=first_text_seen,
                started_at=started_at,
            )

        self._record_event(
            event=FinishEvent(reason="stop"),
            result=result,
            on_event=on_event,
            first_text_seen=first_text_seen,
            started_at=started_at,
        )

    def run_turn(
        self,
        user_input,
        active_target,
        conversation,
        *,
        stop_sequences: Optional[List[str]] = None,
        on_event: Optional[Callable[[ModelEvent], None]] = None,
        on_wait: Optional[Callable[[int, int, int], None]] = None,
        on_retry: Optional[Callable[[int, int, str], None]] = None,
    ) -> AssistantTurnResult:
        """Run one generation turn and return structured output."""
        flags = self._tooling_flags()
        tools_enabled = flags["enabled"]
        local_tools_experimental = flags["local_mode"] == "experimental"
        max_iterations = flags["max_iterations"]

        profile = flags["profile"] if tools_enabled else "off"
        registry = ToolRegistry(repo_root=Path.cwd(), profile=profile)
        session_id = conversation.conversation_id if conversation else "session"

        result = AssistantTurnResult(text="")
        started_at = time.time()
        first_text_seen: Dict[str, float] = {}

        if active_target.backend == "cloud":
            cloud_messages = self._build_message_window(conversation=conversation)
            if not cloud_messages:
                cloud_messages = [{"role": "user", "content": user_input}]

            if tools_enabled:
                cloud_messages = [
                    {
                        "role": "system",
                        "content": TOOLS_SYSTEM_INSTRUCTION_TEMPLATE.format(cwd=str(Path.cwd())),
                    },
                    *cloud_messages,
                ]
            else:
                cloud_messages = [{"role": "system", "content": NO_TOOLS_SYSTEM_INSTRUCTION}, *cloud_messages]

            tool_specs = registry.specs() if tools_enabled else []

            def execute_tool(call: ToolCall) -> ToolResult:
                return self._execute_tool_call(
                    registry=registry,
                    session_id=session_id,
                    call=call,
                )

            events = self.cli.cloud_router.stream_events(
                model_ref=active_target.cloud_model,
                messages=cloud_messages,
                max_tokens=self.cli.config.inference.max_tokens,
                temperature=self.cli.config.inference.temperature,
                top_p=self.cli.config.inference.top_p,
                tools=tool_specs,
                tool_choice="auto",
                tool_executor=execute_tool if tools_enabled else None,
                on_wait=on_wait,
                on_retry=on_retry,
            )

            for event in events:
                self._record_event(
                    event=event,
                    result=result,
                    on_event=on_event,
                    first_text_seen=first_text_seen,
                    started_at=started_at,
                )

        else:
            self._run_local(
                user_input=user_input,
                conversation=conversation,
                registry=registry,
                tools_enabled=tools_enabled,
                local_tools_experimental=local_tools_experimental,
                max_iterations=max_iterations,
                stop_sequences=stop_sequences,
                on_event=on_event,
                result=result,
                first_text_seen=first_text_seen,
                started_at=started_at,
            )

        result.elapsed_seconds = time.time() - started_at
        result.first_token_latency_seconds = first_text_seen.get("first")
        return result
