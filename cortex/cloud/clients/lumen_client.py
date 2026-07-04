"""Client for the managed local Lumen server (OpenAI Chat Completions wire).

Lumen implements `/v1/chat/completions` with SSE streaming and native tool
calls (verified against lumen v0.2.0/v0.3.0). The cloud OpenAI client speaks
the newer Responses API, which Lumen does not serve, so local turns use this
dedicated client. It emits the same normalized event stream as the cloud
clients (TextDeltaEvent / ToolCallEvent / ToolResultEvent / FinishEvent).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from openai import OpenAI

from cortex.tooling.types import (
    FinishEvent,
    TextDeltaEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
)

logger = logging.getLogger(__name__)


class LumenClient:
    """OpenAI-compatible chat client bound to the local lumen-server."""

    def __init__(self, *, base_url: str, timeout_seconds: int = 600):
        self.base_url = base_url
        self.client = OpenAI(api_key="lumen", base_url=base_url, timeout=timeout_seconds)

    def _provenance(self, reported_model: str, response_id: str) -> Dict[str, object]:
        """Response-side identity proof for post-turn verification."""
        return {
            "client_kind": "lumen",
            "reported_model": reported_model,
            "response_id": response_id[:40],
            "endpoint": self.base_url,
        }

    def validate_key(self) -> Tuple[bool, str]:
        return True, "local"

    @staticmethod
    def _serialize_tools(tools) -> List[Dict[str, object]]:
        serialized: List[Dict[str, object]] = []
        for tool in tools or []:
            serialized.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return serialized

    @staticmethod
    def _format_tool_output(result: ToolResult) -> str:
        # metadata.diff is a UI-only payload (up to 400 rows) — never feed it
        # back into the model's context.
        metadata = {k: v for k, v in (result.metadata or {}).items() if k != "diff"}
        payload = {
            "ok": result.ok,
            "output": result.output,
            "error": result.error,
            "metadata": metadata,
        }
        return json.dumps(payload, ensure_ascii=True)

    def stream_events(
        self,
        *,
        model_id: str,
        messages: Iterable[Dict[str, object]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools=None,
        tool_choice: str = "auto",  # accepted for interface parity; Lumen is always auto
        tool_executor=None,
        max_tool_iterations: int = 8,
    ):
        """Yield normalized events from a (possibly multi-hop) tool loop."""
        conversation: List[Dict[str, object]] = [dict(message) for message in messages]
        serialized_tools = self._serialize_tools(tools)
        use_tools = bool(serialized_tools) and tool_executor is not None
        reported_model = ""
        response_id = ""

        for _ in range(max(1, max_tool_iterations)):
            kwargs: Dict[str, object] = {
                "model": model_id,
                "messages": conversation,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,
            }
            if use_tools:
                # Lumen's strict wire schema accepts `tools` but rejects
                # `tool_choice` (400 unknown_field) — auto is its behavior.
                kwargs["tools"] = serialized_tools

            stream = self.client.chat.completions.create(**cast(Any, kwargs))

            text_parts: List[str] = []
            pending: Dict[int, Dict[str, str]] = {}
            finish_reason: Optional[str] = None

            for chunk in stream:
                chunk_model = getattr(chunk, "model", None)
                if chunk_model:
                    reported_model = str(chunk_model)
                chunk_id = getattr(chunk, "id", None)
                if chunk_id:
                    response_id = str(chunk_id)
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    content = getattr(delta, "content", None)
                    if content:
                        text_parts.append(str(content))
                        yield TextDeltaEvent(delta=str(content))
                    for tool_delta in getattr(delta, "tool_calls", None) or []:
                        index = int(getattr(tool_delta, "index", 0) or 0)
                        slot = pending.setdefault(index, {"id": "", "name": "", "args": ""})
                        if getattr(tool_delta, "id", None):
                            slot["id"] = str(tool_delta.id)
                        function = getattr(tool_delta, "function", None)
                        if function is not None:
                            if getattr(function, "name", None):
                                slot["name"] = str(function.name)
                            if getattr(function, "arguments", None):
                                slot["args"] += str(function.arguments)
                if getattr(choice, "finish_reason", None):
                    finish_reason = str(choice.finish_reason)

            if not pending or not use_tools:
                yield FinishEvent(
                    reason=finish_reason or "stop",
                    provenance=self._provenance(reported_model, response_id),
                )
                return

            # Feed the assistant's tool calls back, execute each, and continue.
            ordered = sorted(pending.items())
            conversation.append(
                {
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                    "tool_calls": [
                        {
                            "id": slot["id"] or f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": slot["name"],
                                "arguments": slot["args"] or "{}",
                            },
                        }
                        for index, slot in ordered
                    ],
                }
            )
            for index, slot in ordered:
                raw_args = slot["args"] or "{}"
                try:
                    parsed = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed = {"_raw": raw_args}
                arguments = parsed if isinstance(parsed, dict) else {"_raw": raw_args}
                call = ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=arguments,
                )
                yield ToolCallEvent(call=call)
                result = tool_executor(call)
                yield ToolResultEvent(result=result)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": self._format_tool_output(result),
                    }
                )

        raise RuntimeError("Lumen tool loop exceeded max iterations")
