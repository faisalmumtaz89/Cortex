"""OpenAI cloud client wrapper."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from openai import OpenAI

from cortex.cloud.retry import RetryCallback, ServerError, StreamCutOffError, with_retries
from cortex.cloud.types import OPENAI_BASE_URL
from cortex.tooling.types import (
    FinishEvent,
    TextDeltaEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    TurnInterrupt,
)


class OpenAIClient:
    """OpenAI Responses API client.

    Also serves OpenAI-compatible endpoints (Azure OpenAI's /openai/v1/ path)
    via base_url.
    """

    # Class-level defaults: some tests construct via __new__ and skip __init__.
    endpoint: str = OPENAI_BASE_URL

    def __init__(self, api_key: str, timeout_seconds: int = 60, base_url: Optional[str] = None):
        # Retries are Cortex's own (cortex.cloud.retry), not the SDK's. The
        # host is always given, so the SDK never reads OPENAI_BASE_URL.
        self.endpoint = base_url or OPENAI_BASE_URL
        self.client = OpenAI(
            api_key=api_key, timeout=timeout_seconds, base_url=self.endpoint, max_retries=0
        )

    def _provenance(self, reported_model: object, response_id: object) -> Dict[str, object]:
        """Response-side identity proof for post-turn verification."""
        record: Dict[str, object] = {
            "client_kind": "openai",
            "reported_model": str(reported_model or ""),
            "response_id": str(response_id or "")[:40],
            "endpoint": self.endpoint,
        }
        return record

    def _normalize_messages(
        self, messages: Iterable[Dict[str, object]]
    ) -> Tuple[Optional[str], List[Dict[str, object]]]:
        """Split off the system text and convert Chat Completions messages to
        Responses input items: text messages, function calls and their
        outputs."""
        system_parts: List[str] = []
        normalized: List[Dict[str, object]] = []

        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content") or "").strip()
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            if role == "tool":
                normalized.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.get("tool_call_id", "")),
                        "output": str(message.get("content") or ""),
                    }
                )
                continue
            if role not in {"user", "assistant"}:
                continue
            if content:
                normalized.append({"role": role, "content": content})
            for call in cast(List[Dict[str, Any]], message.get("tool_calls") or []):
                normalized.append(
                    {
                        "type": "function_call",
                        "call_id": str(call["id"]),
                        "name": str(call["function"]["name"]),
                        "arguments": str(call["function"]["arguments"]),
                    }
                )

        system_text = "\n\n".join(system_parts).strip() or None
        return system_text, normalized

    @staticmethod
    def _item_get(item: object, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _extract_output_text(self, response_obj: object) -> str:
        """Best-effort extraction of text from a final Responses API object."""
        if response_obj is None:
            return ""

        output_text = getattr(response_obj, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        if isinstance(output_text, list):
            parts = [str(part) for part in output_text if str(part).strip()]
            if parts:
                return "".join(parts)

        output_items = getattr(response_obj, "output", None)
        if isinstance(output_items, list):
            collected: List[str] = []
            for item in output_items:
                content = self._item_get(item, "content", None)
                if not isinstance(content, list):
                    continue
                for part in content:
                    part_text = self._item_get(part, "text", None)
                    if isinstance(part_text, str) and part_text:
                        collected.append(part_text)
            if collected:
                return "".join(collected)

        return ""

    def _serialize_tools(self, tools) -> List[Dict[str, object]]:
        serialized: List[Dict[str, object]] = []
        for tool in tools or []:
            serialized.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            )
        return serialized

    def _extract_tool_calls(self, response_obj: object) -> List[ToolCall]:
        calls: List[ToolCall] = []
        output_items = getattr(response_obj, "output", None)
        if not isinstance(output_items, list):
            return calls

        for item in output_items:
            item_type = str(self._item_get(item, "type", "")).strip().lower()
            if item_type not in {"function_call", "tool_call"}:
                continue

            call_id = self._item_get(item, "call_id", None) or self._item_get(item, "id", None)
            name = self._item_get(item, "name", None)
            if not name:
                function_obj = self._item_get(item, "function", None)
                name = self._item_get(function_obj, "name", None)

            raw_args = self._item_get(item, "arguments", None)
            if raw_args is None:
                function_obj = self._item_get(item, "function", None)
                raw_args = self._item_get(function_obj, "arguments", {})

            parsed_args: Dict[str, object]
            if isinstance(raw_args, str):
                try:
                    loaded = json.loads(raw_args) if raw_args.strip() else {}
                    parsed_args = loaded if isinstance(loaded, dict) else {}
                except Exception:
                    parsed_args = {}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            else:
                parsed_args = {}

            if not isinstance(name, str) or not name.strip():
                continue

            if not isinstance(call_id, str) or not call_id.strip():
                call_id = f"call_{len(calls) + 1}"

            calls.append(ToolCall(id=call_id, name=name, arguments=parsed_args))

        return calls

    def _format_tool_output(self, result: ToolResult) -> str:
        payload = {
            "ok": result.ok,
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
        }
        return json.dumps(payload, ensure_ascii=True)

    def _build_response_kwargs(
        self,
        *,
        model_id: str,
        request_input,
        max_tokens: int,
        temperature: float,
        top_p: float,
        system_text: Optional[str],
        tools=None,
        tool_choice: str = "auto",
        previous_response_id: Optional[str] = None,
    ) -> Dict[str, object]:
        include_sampling_controls = self._supports_sampling_controls(model_id)
        kwargs: Dict[str, object] = {
            "model": model_id,
            "input": request_input,
            "max_output_tokens": max_tokens,
        }
        if include_sampling_controls:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        if system_text:
            kwargs["instructions"] = system_text
        if tools:
            kwargs["tools"] = self._serialize_tools(tools)
            kwargs["tool_choice"] = tool_choice
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        return kwargs

    @staticmethod
    def _stream_text(stream, outcome: Dict[str, object]):
        """Yield the text deltas of a Responses stream and record its final
        response in ``outcome["response"]``. The final response is taken from
        the stream's own response.completed / response.incomplete event:
        the SDK's get_final_response() raises for an incomplete response. An
        error event raises ServerError with the server's message."""
        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield TextDeltaEvent(delta=str(delta))
            elif event_type in ("response.completed", "response.incomplete"):
                outcome["response"] = getattr(event, "response", None)
            elif event_type in ("error", "response.failed"):
                error = getattr(getattr(event, "response", None), "error", None) or event
                code = getattr(error, "code", None)
                message = getattr(error, "message", None) or "the response failed"
                raise ServerError(f"{message} ({code})" if code else str(message), code)

    @staticmethod
    def _finish_reason(response: object) -> str:
        """The response's finish reason: length if it stopped at max_output_tokens."""
        details = getattr(response, "incomplete_details", None)
        if (
            getattr(response, "status", None) == "incomplete"
            and getattr(details, "reason", None) == "max_output_tokens"
        ):
            return "length"
        return "stop"

    def _supports_sampling_controls(self, model_id: str) -> bool:
        """Return whether temperature/top_p should be sent for this model."""
        normalized = str(model_id or "").strip().lower()
        if normalized.startswith("gpt-5"):
            return False
        return True

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
        """Yield normalized events from OpenAI responses."""
        interrupt = interrupt or TurnInterrupt()
        system_text, normalized_messages = self._normalize_messages(messages)
        if not normalized_messages:
            raise RuntimeError("No user/assistant messages available for OpenAI request.")
        use_tools = bool(tools) and tool_executor is not None

        request_input: object = normalized_messages
        previous_response_id: Optional[str] = None
        reported_model: object = ""
        reported_id: object = ""

        for _ in range(max(1, max_tool_iterations)):
            kwargs = self._build_response_kwargs(
                model_id=model_id,
                request_input=request_input,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system_text=system_text,
                tools=tools if use_tools else None,
                tool_choice=tool_choice,
                previous_response_id=previous_response_id,
            )
            response, emitted = yield from with_retries(
                lambda: self._request(kwargs, interrupt),
                retries=retries,
                interrupt=interrupt,
                on_retry=on_retry,
            )

            if not emitted:
                text = self._extract_output_text(response)
                if text:
                    yield TextDeltaEvent(delta=text)

            previous_response_id = getattr(response, "id", None) or previous_response_id
            reported_model = getattr(response, "model", None) or reported_model
            reported_id = getattr(response, "id", None) or reported_id
            calls = self._extract_tool_calls(response) if use_tools else []
            # A reply cut off by max_output_tokens ends the turn: its tool
            # calls may be incomplete and must not run.
            finish_reason = self._finish_reason(response)
            if not calls or finish_reason == "length":
                yield FinishEvent(
                    reason=finish_reason,
                    provenance=self._provenance(reported_model, reported_id),
                )
                return

            outputs = []
            for call in calls:
                yield ToolCallEvent(call=call)
                tool_result = tool_executor(call)
                yield ToolResultEvent(result=tool_result)
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.id,
                        "output": self._format_tool_output(tool_result),
                    }
                )

            request_input = outputs

        # The step limit ends the turn; its tool calls and results are kept.
        yield FinishEvent(
            reason="tool_limit", provenance=self._provenance(reported_model, reported_id)
        )

    def _request(self, kwargs: Dict[str, object], interrupt: TurnInterrupt):
        """Send one request and yield its text as it streams; return the
        response and whether any text was streamed."""
        interrupt.raise_if_set()
        outcome: Dict[str, object] = {}
        emitted = False
        with self.client.responses.stream(**cast(Any, kwargs)) as stream:
            with interrupt.watching(stream):
                for text_event in self._stream_text(stream, outcome):
                    emitted = True
                    yield text_event
        response = outcome.get("response")
        if response is None:
            raise StreamCutOffError("The response stream ended before the reply was complete.")
        return response, emitted
