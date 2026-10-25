"""Anthropic cloud client wrapper."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from anthropic import Anthropic

from cortex.cloud.retry import RetryCallback, StreamCutOffError, with_retries
from cortex.tooling.types import (
    FinishEvent,
    TextDeltaEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    TurnInterrupt,
)


class AnthropicClient:
    """Anthropic Messages API client."""

    # Class-level defaults: some tests construct via __new__ and skip __init__.
    endpoint: str = "https://api.anthropic.com"

    def __init__(self, api_key: str, timeout_seconds: int = 60):
        # Retries are Cortex's own (cortex.cloud.retry), not the SDK's.
        self.client = Anthropic(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        self.endpoint = str(getattr(self.client, "base_url", "") or "https://api.anthropic.com")

    def _provenance(self, reported_model: Any, response_id: Any) -> Dict[str, Any]:
        """Response-side identity proof for post-turn verification."""
        record: Dict[str, Any] = {
            "client_kind": "anthropic",
            "reported_model": str(reported_model or ""),
            "response_id": str(response_id or "")[:40],
            "endpoint": self.endpoint,
        }
        return record

    @staticmethod
    def _finish_reason(stop_reason: object) -> str:
        """Map a Messages API stop_reason onto the turn's finish reason."""
        if stop_reason == "max_tokens":
            return "length"
        if stop_reason == "model_context_window_exceeded":
            return "context_window"
        return "stop"

    @staticmethod
    def _item_get(item: object, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _normalize_messages(
        self, messages: Iterable[Dict[str, object]]
    ) -> Tuple[Optional[str], List[Dict[str, object]]]:
        """Split off the system text and convert Chat Completions messages to
        Messages API turns: tool calls become tool_use blocks and their results
        tool_result blocks in the following user turn."""
        system_parts: List[str] = []
        normalized: List[Dict[str, object]] = []
        results: Optional[List[Dict[str, object]]] = None

        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content") or "").strip()
            if role != "tool":
                results = None
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            if role == "tool":
                if results is None:
                    results = []
                    normalized.append({"role": "user", "content": results})
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(message.get("tool_call_id", "")),
                        "content": str(message.get("content") or ""),
                    }
                )
                continue
            if role not in {"user", "assistant"}:
                continue
            blocks: List[Dict[str, object]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for call in cast(List[Dict[str, Any]], message.get("tool_calls") or []):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(call["id"]),
                        "name": str(call["function"]["name"]),
                        "input": json.loads(call["function"]["arguments"] or "{}"),
                    }
                )
            if blocks:
                normalized.append({"role": role, "content": blocks})

        system_text = "\n\n".join(system_parts).strip() or None
        return system_text, normalized

    def _serialize_tools(self, tools) -> List[Dict[str, object]]:
        serialized: List[Dict[str, object]] = []
        for tool in tools or []:
            serialized.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
            )
        return serialized

    def _serialize_content_block(self, block: object) -> Dict[str, object]:
        block_type = str(self._item_get(block, "type", "")).strip().lower()
        if block_type == "text":
            return {"type": "text", "text": str(self._item_get(block, "text", ""))}
        if block_type == "tool_use":
            return {
                "type": "tool_use",
                "id": str(self._item_get(block, "id", "")),
                "name": str(self._item_get(block, "name", "")),
                "input": self._item_get(block, "input", {}),
            }
        if block_type == "thinking":
            # Claude 5 family emits thinking blocks by default on tool-use
            # turns. Replay them VERBATIM: `thinking` is REQUIRED even when
            # empty (verified live — a block can stream only a signature_delta,
            # thinking=""), and the signature authenticates the block. Dropping
            # the field 400s: "content.0.thinking.thinking: Field required".
            return {
                "type": "thinking",
                "thinking": str(self._item_get(block, "thinking", "") or ""),
                "signature": str(self._item_get(block, "signature", "") or ""),
            }
        if block_type == "redacted_thinking":
            # Opaque encrypted reasoning — pass through untouched.
            return {
                "type": "redacted_thinking",
                "data": str(self._item_get(block, "data", "") or ""),
            }
        # Unknown server block: replay its own shape verbatim when possible
        # rather than fabricating a text block the API may reject.
        dump = getattr(block, "model_dump", None)
        if callable(dump):
            try:
                raw = dump()
                if isinstance(raw, dict) and raw.get("type"):
                    return raw
            except Exception:  # pragma: no cover - SDK-specific edge
                pass
        if isinstance(block, dict) and block.get("type"):
            return dict(block)
        return {"type": block_type or "text", "text": str(self._item_get(block, "text", ""))}

    def stream_events(
        self,
        *,
        model_id: str,
        messages: Iterable[Dict[str, object]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools=None,
        tool_choice: str = "auto",
        tool_executor=None,
        max_tool_iterations: int = 100,
        interrupt: Optional[TurnInterrupt] = None,
        retries: int = 0,
        on_retry: Optional[RetryCallback] = None,
    ):
        """Yield normalized events from Anthropic messages API."""
        interrupt = interrupt or TurnInterrupt()
        system_text, normalized_messages = self._normalize_messages(messages)
        if not normalized_messages:
            raise RuntimeError("No user/assistant messages available for Anthropic request.")
        use_tools = bool(tools) and tool_executor is not None

        history = list(normalized_messages)
        reported_model: Any = ""
        reported_id: Any = ""
        for _ in range(max(1, max_tool_iterations)):
            # No temperature/top_p: Claude-5-era models 400 on temperature
            # ("deprecated for this model") and 4-x-era models 400 when both
            # are sent — omitting both works on every generation (verified
            # live against fable-5/sonnet-5/opus-4-8/haiku-4-5/opus-4-6).
            kwargs: Dict[str, Any] = {
                "model": model_id,
                "messages": history,
                "max_tokens": max_tokens,
            }
            if system_text:
                kwargs["system"] = system_text
            if use_tools:
                kwargs["tools"] = self._serialize_tools(tools)

            response, emitted = yield from with_retries(
                lambda: self._request(kwargs, interrupt),
                retries=retries,
                interrupt=interrupt,
                on_retry=on_retry,
            )

            reported_model = self._item_get(response, "model", None) or reported_model
            reported_id = self._item_get(response, "id", None) or reported_id

            content = self._item_get(response, "content", [])
            if not isinstance(content, list):
                content = []

            text_parts: List[str] = []
            tool_calls: List[ToolCall] = []
            assistant_blocks: List[Dict[str, object]] = []

            for block in content:
                assistant_blocks.append(self._serialize_content_block(block))
                block_type = str(self._item_get(block, "type", "")).strip().lower()
                if block_type == "text":
                    text = self._item_get(block, "text", "")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                elif block_type == "tool_use" and use_tools:
                    tool_id = str(self._item_get(block, "id", "") or f"call_{len(tool_calls) + 1}")
                    name = str(self._item_get(block, "name", "")).strip()
                    raw_input = self._item_get(block, "input", {})
                    arguments = raw_input if isinstance(raw_input, dict) else {}
                    if name:
                        tool_calls.append(ToolCall(id=tool_id, name=name, arguments=arguments))

            if text_parts and not emitted:
                yield TextDeltaEvent(delta="".join(text_parts))

            # A reply cut off by a limit ends the turn: its tool calls may
            # be incomplete and must not run.
            finish_reason = self._finish_reason(self._item_get(response, "stop_reason", None))
            if not tool_calls or finish_reason != "stop":
                yield FinishEvent(
                    reason=finish_reason,
                    provenance=self._provenance(reported_model, reported_id),
                )
                return

            tool_result_blocks: List[Dict[str, object]] = []
            for call in tool_calls:
                yield ToolCallEvent(call=call)
                result: ToolResult = tool_executor(call)
                yield ToolResultEvent(result=result)

                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(
                            {
                                "ok": result.ok,
                                "output": result.output,
                                "error": result.error,
                                "metadata": result.metadata,
                            },
                            ensure_ascii=True,
                        ),
                    }
                )

            history.append({"role": "assistant", "content": assistant_blocks})
            history.append({"role": "user", "content": tool_result_blocks})

        # The step limit ends the turn; its tool calls and results are kept.
        yield FinishEvent(
            reason="tool_limit", provenance=self._provenance(reported_model, reported_id)
        )

    def _request(self, kwargs: Dict[str, Any], interrupt: TurnInterrupt):
        """Send one request and yield its text as it streams; return the final
        message and whether any text was streamed."""
        interrupt.raise_if_set()
        emitted = False
        with self.client.messages.stream(**kwargs) as stream:
            with interrupt.watching(stream):
                for text in stream.text_stream:
                    if text:
                        emitted = True
                        yield TextDeltaEvent(delta=str(text))
                response = stream.get_final_message()
        if self._item_get(response, "stop_reason", None) is None:
            raise StreamCutOffError("The response stream ended before the reply was complete.")
        return response, emitted
