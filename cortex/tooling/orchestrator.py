"""Tool-aware generation orchestrator."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, cast

from cortex.cloud.types import CloudModelRef, CloudProvider
from cortex.lumen_runtime import parse_selector
from cortex.tooling.agent_prompt import build_system_prompt
from cortex.tooling.permissions import (
    PermissionDecision,
    PermissionDeniedError,
    PermissionManager,
    PermissionRequest,
    default_rules,
)
from cortex.tooling.provenance import verify_turn_provenance
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


class ToolingOrchestrator:
    """Coordinates generation, tool execution, and permission checks."""

    def __init__(self, *, cli):
        self.cli = cli
        self.permission_manager = PermissionManager(rules=default_rules())

    def _tooling_flags(self) -> dict:
        tools_cfg = getattr(self.cli.config, "tools", None)
        tools_enabled = bool(getattr(tools_cfg, "tools_enabled", False))
        tools_profile = str(getattr(tools_cfg, "tools_profile", "off") or "off")
        tools_local_mode = str(getattr(tools_cfg, "tools_local_mode", "disabled") or "disabled")
        max_iterations = int(getattr(tools_cfg, "tools_max_iterations", 25) or 25)
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
            if event.provenance is not None:
                result.provenance = dict(event.provenance)
            return

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
        max_iterations = flags["max_iterations"]

        profile = flags["profile"] if tools_enabled else "off"
        registry = ToolRegistry(repo_root=Path.cwd(), profile=profile)
        session_id = conversation.conversation_id if conversation else "session"

        result = AssistantTurnResult(text="")
        started_at = time.time()
        first_text_seen: Dict[str, float] = {}

        local_selector: Optional[str] = None
        if active_target.backend == "cloud":
            model_ref = active_target.cloud_model
        else:
            # Local models are served by the managed Lumen engine and speak the
            # same OpenAI-compatible protocol, so both backends share one loop.
            selector = active_target.local_model
            if not selector:
                raise RuntimeError("No model loaded. Pick one with /model.")
            ok, message = self.cli.lumen_runtime.ensure_server(selector)
            if not ok:
                raise RuntimeError(f"local · {selector} failed to start: {message}")
            local_selector = selector
            model_ref = CloudModelRef(
                provider=CloudProvider.LUMEN, model_id=parse_selector(selector)[0]
            )

        messages = self._build_message_window(conversation=conversation)
        if not messages:
            messages = [{"role": "user", "content": user_input}]

        if tools_enabled:
            messages = [
                {
                    "role": "system",
                    "content": build_system_prompt(cwd=registry.repo_root),
                },
                *messages,
            ]
        else:
            messages = [{"role": "system", "content": NO_TOOLS_SYSTEM_INSTRUCTION}, *messages]

        tool_specs = registry.specs() if tools_enabled else []

        def execute_tool(call: ToolCall) -> ToolResult:
            return self._execute_tool_call(
                registry=registry,
                session_id=session_id,
                call=call,
            )

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

        self._verify_turn_provenance(
            result=result,
            model_ref=model_ref,
            local_selector=local_selector,
            on_event=on_event,
        )

        result.elapsed_seconds = time.time() - started_at
        result.first_token_latency_seconds = first_text_seen.get("first")
        return result

    def _verify_turn_provenance(
        self,
        *,
        result: AssistantTurnResult,
        model_ref: CloudModelRef,
        local_selector: Optional[str],
        on_event: Optional[Callable[[ModelEvent], None]],
    ) -> None:
        """Fail the turn unless the response proved it came from the
        requested model (see cortex/tooling/provenance.py)."""
        is_local = model_ref.provider == CloudProvider.LUMEN
        expected_endpoint: Optional[str] = None
        lumen_ready: Optional[bool] = None
        if is_local:
            runtime = self.cli.lumen_runtime
            expected_endpoint = runtime.base_url()
            lumen_ready = bool(runtime.status().get("ready"))

        verdict = verify_turn_provenance(
            provider=model_ref.provider,
            requested_model=model_ref.model_id,
            provenance=result.provenance,
            expected_endpoint=expected_endpoint,
            lumen_ready=lumen_ready,
        )

        intent_label = (
            f"local · {local_selector}" if is_local else f"cloud · {model_ref.selector}"
        )
        if not verdict.ok:
            error = (
                f"Model provenance mismatch: asked {intent_label}, but {verdict.reason} "
                f"— turn rejected."
            )
            if on_event is not None:
                on_event(ErrorEvent(error=error, metadata={"provenance": result.provenance or {}}))
            raise RuntimeError(error)

        result.provenance_verified = True
        result.served_backend = "local" if is_local else "cloud"
        served = local_selector if is_local else model_ref.selector
        result.served_model_label = f"{served} (scripted)" if verdict.scripted else str(served)
