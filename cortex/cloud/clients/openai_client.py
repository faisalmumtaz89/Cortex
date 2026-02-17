"""OpenAI cloud client wrapper."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from cortex.tooling.types import (
    FinishEvent,
    TextDeltaEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
)

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI streaming and key validation wrapper."""

    def __init__(self, api_key: str, timeout_seconds: int = 60):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "OpenAI runtime dependency is missing. Run /login openai to auto-install and configure it."
            ) from exc

        self.client = OpenAI(api_key=api_key, timeout=timeout_seconds)

    def validate_key(self) -> Tuple[bool, str]:
        """Validate API key using a low-cost API call."""
        try:
            self.client.models.list()
            return True, "OpenAI API key is valid."
        except Exception as exc:
            return False, f"OpenAI authentication failed: {exc}"

    def _normalize_messages(self, messages: Iterable[Dict[str, object]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
        system_parts: List[str] = []
        normalized: List[Dict[str, str]] = []

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
            normalized.append({"role": role, "content": content})

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
        max_tool_iterations: int = 8,
    ):
        """Yield normalized events from OpenAI responses."""
        system_text, normalized_messages = self._normalize_messages(messages)
        if not normalized_messages:
            raise RuntimeError("No user/assistant messages available for OpenAI request.")

        if tools and tool_executor is not None:
            request_input: object = normalized_messages
            previous_response_id: Optional[str] = None

            for _ in range(max_tool_iterations):
                kwargs = self._build_response_kwargs(
                    model_id=model_id,
                    request_input=request_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    system_text=system_text,
                    tools=tools,
                    tool_choice=tool_choice,
                    previous_response_id=previous_response_id,
                )
                emitted_this_turn = False
                response = None

                try:
                    with self.client.responses.stream(**cast(Any, kwargs)) as stream:
                        if hasattr(stream, "text_deltas"):
                            for delta in stream.text_deltas:
                                if delta:
                                    emitted_this_turn = True
                                    yield TextDeltaEvent(delta=str(delta))
                        else:
                            for event in stream:
                                event_type = getattr(event, "type", "")
                                if event_type == "response.output_text.delta":
                                    delta = getattr(event, "delta", "")
                                    if delta:
                                        emitted_this_turn = True
                                        yield TextDeltaEvent(delta=str(delta))

                        if hasattr(stream, "get_final_response"):
                            response = stream.get_final_response()
                except Exception as exc:
                    logger.debug(
                        "OpenAI Responses API tool stream failed, falling back to non-streaming create: %s",
                        exc,
                    )

                if response is None:
                    response = self.client.responses.create(**cast(Any, kwargs))

                if not emitted_this_turn:
                    text = self._extract_output_text(response)
                    if text:
                        yield TextDeltaEvent(delta=text)

                previous_response_id = getattr(response, "id", None) or previous_response_id
                calls = self._extract_tool_calls(response)
                if not calls:
                    yield FinishEvent(reason="stop")
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

            raise RuntimeError("OpenAI tool loop exceeded max iterations")

        response_kwargs = self._build_response_kwargs(
            model_id=model_id,
            request_input=normalized_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system_text=system_text,
        )

        try:
            with self.client.responses.stream(**cast(Any, response_kwargs)) as stream:
                emitted = False

                if hasattr(stream, "text_deltas"):
                    for delta in stream.text_deltas:
                        if delta:
                            emitted = True
                            yield TextDeltaEvent(delta=str(delta))
                else:
                    for event in stream:
                        event_type = getattr(event, "type", "")
                        if event_type == "response.output_text.delta":
                            delta = getattr(event, "delta", "")
                            if delta:
                                emitted = True
                                yield TextDeltaEvent(delta=str(delta))

                if not emitted and hasattr(stream, "get_final_response"):
                    final_response = stream.get_final_response()
                    final_text = self._extract_output_text(final_response)
                    if final_text:
                        yield TextDeltaEvent(delta=final_text)

            yield FinishEvent(reason="stop")
            return
        except Exception as exc:
            logger.debug("OpenAI Responses API stream failed, falling back to chat completions: %s", exc)

        completion_messages = list(normalized_messages)
        if system_text:
            completion_messages = [{"role": "system", "content": system_text}] + completion_messages

        completion_kwargs = {
            "model": model_id,
            "messages": completion_messages,
            "stream": True,
        }
        if self._supports_sampling_controls(model_id):
            completion_kwargs["temperature"] = temperature
            completion_kwargs["top_p"] = top_p

        try:
            stream = self.client.chat.completions.create(
                max_completion_tokens=max_tokens,
                **cast(Any, completion_kwargs),
            )
        except TypeError:
            stream = self.client.chat.completions.create(
                max_tokens=max_tokens,
                **cast(Any, completion_kwargs),
            )

        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except Exception:
                delta = None
            if delta:
                yield TextDeltaEvent(delta=str(delta))

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
            raise RuntimeError("No user/assistant messages available for OpenAI request.")

        response_kwargs = self._build_response_kwargs(
            model_id=model_id,
            request_input=normalized_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system_text=system_text,
        )

        response = self.client.responses.create(**cast(Any, response_kwargs))
        text = self._extract_output_text(response)
        if text:
            return text

        completion_messages = list(normalized_messages)
        if system_text:
            completion_messages = [{"role": "system", "content": system_text}] + completion_messages
        try:
            completion = self.client.chat.completions.create(
                model=model_id,
                messages=cast(Any, completion_messages),
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_tokens,
            )
        except TypeError:
            completion = self.client.chat.completions.create(
                model=model_id,
                messages=cast(Any, completion_messages),
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        try:
            content = completion.choices[0].message.content
        except Exception:
            content = None
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text:
                    parts.append(part_text)
            if parts:
                return "".join(parts)

        raise RuntimeError("OpenAI non-stream fallback returned no text.")
