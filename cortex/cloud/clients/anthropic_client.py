"""Anthropic cloud client wrapper."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cortex.tooling.types import (
    FinishEvent,
    TextDeltaEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
)

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Anthropic streaming and key validation wrapper."""

    def __init__(self, api_key: str, timeout_seconds: int = 60):
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Anthropic runtime dependency is missing. Run /login anthropic to auto-install and configure it."
            ) from exc

        self.client = Anthropic(api_key=api_key, timeout=timeout_seconds)

    def validate_key(self) -> Tuple[bool, str]:
        """Validate API key using models list call."""
        try:
            self.client.models.list(limit=1)
            return True, "Anthropic API key is valid."
        except Exception as exc:
            return False, f"Anthropic authentication failed: {exc}"

    @staticmethod
    def _item_get(item: object, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _normalize_messages(
        self, messages: Iterable[Dict[str, object]]
    ) -> Tuple[Optional[str], List[Dict[str, object]]]:
        system_parts: List[str] = []
        normalized: List[Dict[str, object]] = []

        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if not content:
                continue

            if role == "system":
                system_parts.append(content)
                continue

            if role not in {"user", "assistant"}:
                continue

            normalized.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                }
            )

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
        max_tool_iterations: int = 8,
    ):
        """Yield normalized events from Anthropic messages API."""
        system_text, normalized_messages = self._normalize_messages(messages)
        if not normalized_messages:
            raise RuntimeError("No user/assistant messages available for Anthropic request.")

        if tools and tool_executor is not None:
            history = list(normalized_messages)
            for _ in range(max_tool_iterations):
                kwargs: Dict[str, object] = {
                    "model": model_id,
                    "messages": history,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if top_p is not None:
                    kwargs["top_p"] = top_p
                if system_text:
                    kwargs["system"] = system_text
                kwargs["tools"] = self._serialize_tools(tools)

                emitted_this_turn = False
                response = None
                try:
                    with self.client.messages.stream(**kwargs) as stream:
                        for text in stream.text_stream:
                            if text:
                                emitted_this_turn = True
                                yield TextDeltaEvent(delta=str(text))
                        if hasattr(stream, "get_final_message"):
                            response = stream.get_final_message()
                except Exception as exc:
                    logger.debug(
                        "Anthropic Messages API tool stream failed, falling back to non-streaming create: %s",
                        exc,
                    )

                if response is None:
                    response = self.client.messages.create(**kwargs)

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
                    elif block_type == "tool_use":
                        tool_id = str(self._item_get(block, "id", "") or f"call_{len(tool_calls) + 1}")
                        name = str(self._item_get(block, "name", "")).strip()
                        raw_input = self._item_get(block, "input", {})
                        arguments = raw_input if isinstance(raw_input, dict) else {}
                        if name:
                            tool_calls.append(ToolCall(id=tool_id, name=name, arguments=arguments))

                if text_parts and not emitted_this_turn:
                    yield TextDeltaEvent(delta="".join(text_parts))

                if not tool_calls:
                    yield FinishEvent(reason="stop")
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

            raise RuntimeError("Anthropic tool loop exceeded max iterations")

        final_kwargs: Dict[str, object] = {
            "model": model_id,
            "messages": normalized_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            final_kwargs["top_p"] = top_p
        if system_text:
            final_kwargs["system"] = system_text

        with self.client.messages.stream(**final_kwargs) as stream:
            for text in stream.text_stream:
                if text:
                    yield TextDeltaEvent(delta=str(text))

        yield FinishEvent(reason="stop")

    def stream(
        self,
        *,
        model_id: str,
        messages: Iterable[Dict[str, object]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """Backward-compatible text-only stream."""
        for event in self.stream_events(
            model_id=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=None,
            tool_choice="auto",
            tool_executor=None,
        ):
            if isinstance(event, TextDeltaEvent) and event.delta:
                yield event.delta

    def generate_once(
        self,
        *,
        model_id: str,
        messages: Iterable[Dict[str, object]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Non-stream fallback generation for reliability."""
        system_text, normalized_messages = self._normalize_messages(messages)
        if not normalized_messages:
            raise RuntimeError("No user/assistant messages available for Anthropic request.")

        kwargs: Dict[str, object] = {
            "model": model_id,
            "messages": normalized_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if system_text:
            kwargs["system"] = system_text

        response = self.client.messages.create(**kwargs)
        content = getattr(response, "content", None)
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if str(self._item_get(item, "type", "")).strip().lower() != "text":
                    continue
                text = self._item_get(item, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
            if parts:
                return "".join(parts)

        raise RuntimeError("Anthropic non-stream fallback returned no text.")
