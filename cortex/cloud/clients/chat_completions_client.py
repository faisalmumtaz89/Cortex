"""Client for any endpoint that serves the OpenAI Chat Completions API.

Streams `/chat/completions` over SSE with native tool calls and runs the tool
loop locally. It emits the same normalized event stream as the other clients
(TextDeltaEvent / ToolCallEvent / ToolResultEvent / FinishEvent), plus a
ReasoningEvent for each step whose reasoning the model streams.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, cast

import openai
from openai import OpenAI

from cortex.cloud.retry import RetryCallback, StreamCutOffError, with_retries
from cortex.tooling.types import (
    FinishEvent,
    ReasoningEvent,
    TextDeltaEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    TurnInterrupt,
)

# Delta fields that carry a model's reasoning: DeepSeek's API uses
# reasoning_content; vLLM and OpenRouter-style gateways use reasoning.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


logger = logging.getLogger(__name__)

_CUT_OFFS_THAT_END_A_TURN = 3
# What the model is told when its reply stopped at the output limit: after
# what it had written, which is kept, or when all of it was reasoning, which
# is not.
_CUT_OFF_NOTE = (
    "Your reply was cut off at the output limit before it finished. Continue from where it "
    "stopped. A tool call that was cut off did not run: make it again in full."
)
_NOTHING_WRITTEN_NOTE = (
    "Your reply reached the output limit before you wrote anything: all of it was reasoning, "
    "and none of it was kept. Reason more briefly, then answer or make your tool call."
)

class ChatCompletionsClient:
    """Chat Completions client bound to one base URL."""

    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: int):
        self.base_url = base_url
        self.endpoint = base_url
        # Retries are Cortex's own (cortex.cloud.retry), not the SDK's.
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0
        )

    def _provenance(self, reported_model: str, response_id: str) -> Dict[str, object]:
        """Response-side identity proof for post-turn verification."""
        return {
            "client_kind": "chat_completions",
            "reported_model": reported_model,
            "response_id": response_id[:40],
            "endpoint": self.base_url,
        }

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
        reasoning_effort: Optional[str] = None,
    ):
        """Yield normalized events from a (possibly multi-hop) tool loop."""
        interrupt = interrupt or TurnInterrupt()
        conversation: List[Dict[str, object]] = [dict(message) for message in messages]
        serialized_tools = self._serialize_tools(tools)
        use_tools = bool(serialized_tools) and tool_executor is not None
        step = _Step()
        cut_off = 0  # replies in a row that stopped at the output limit
        images_refused = False  # the server refused a request that had images

        requests_allowed = max(1, max_tool_iterations)
        for request_number in range(1, requests_allowed + 1):
            kwargs: Dict[str, object] = {
                "model": model_id,
                "messages": conversation,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,
            }
            if use_tools:
                kwargs["tools"] = serialized_tools
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

            try:
                step = yield from with_retries(
                    lambda: self._request(kwargs, interrupt),
                    retries=retries,
                    interrupt=interrupt,
                    on_retry=on_retry,
                )
            except openai.APIStatusError as exc:
                refused = exc.status_code in (400, 422)
                if (refused and not images_refused and request_number < requests_allowed
                        and any(_shows_images(message) for message in conversation)):
                    # A server that takes no images (or this one): it is sent
                    # the request again without them, and the rest of the turn
                    # has none.
                    logger.info("A request with images was refused (%s): sending it without them",
                                exc.status_code)
                    images_refused = True
                    conversation[:] = _without_images(conversation)
                    continue
                if not cut_off or exc.status_code not in (400, 422):
                    raise
                # A server that will not take the conversation as continued:
                # the turn ends on the cut-off reply, as it did before.
                logger.info("The continued conversation was refused (%s)", exc.status_code)
                yield FinishEvent(
                    reason="length",
                    provenance=self._provenance(step.reported_model, step.response_id),
                )
                return

            # The reasoning goes back with the step it led to, in this turn and
            # every later one: models that think between tool steps continue
            # from it, and DeepSeek rejects a conversation that omits it.
            if step.reasoning:
                yield ReasoningEvent(field=step.reasoning_field, text=step.reasoning)

            # A reply cut off by max_tokens is told so and goes on: one burst
            # of reasoning must not end a turn that is working. What it wrote
            # is kept; its tool calls may be incomplete and do not run. A
            # model that is cut off again and again is not asked for ever.
            cut_off = cut_off + 1 if step.finish_reason == "length" else 0
            if 0 < cut_off < _CUT_OFFS_THAT_END_A_TURN and request_number < requests_allowed:
                logger.info("A reply stopped at the output limit (%s in a row): continuing", cut_off)
                if step.text:
                    kept: Dict[str, object] = {"role": "assistant", "content": step.text}
                    if step.reasoning:
                        kept[step.reasoning_field] = step.reasoning
                    conversation.append(kept)
                note = _CUT_OFF_NOTE if step.text else _NOTHING_WRITTEN_NOTE
                conversation.append({"role": "user", "content": note})
                continue

            if not step.calls or not use_tools or step.finish_reason == "length":
                yield FinishEvent(
                    reason=step.finish_reason,
                    provenance=self._provenance(step.reported_model, step.response_id),
                )
                return

            # Feed the assistant's tool calls back, execute each, and continue.
            # Some servers send no call ids; the conversation needs unique ones.
            ordered = sorted(step.calls.items())
            for _, slot in ordered:
                slot["id"] = slot["id"] or f"call_{uuid.uuid4().hex[:24]}"
            assistant_step: Dict[str, object] = {
                "role": "assistant",
                "content": step.text or None,
                "tool_calls": [
                    {
                        "id": slot["id"],
                        "type": "function",
                        "function": {"name": slot["name"], "arguments": slot["args"] or "{}"},
                    }
                    for _, slot in ordered
                ],
            }
            if step.reasoning:
                assistant_step[step.reasoning_field] = step.reasoning
            conversation.append(assistant_step)
            shown: List[Dict[str, object]] = []
            for _, slot in ordered:
                raw_args = slot["args"] or "{}"
                try:
                    parsed = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed = {"_raw": raw_args}
                arguments = parsed if isinstance(parsed, dict) else {"_raw": raw_args}
                call = ToolCall(
                    id=slot["id"],
                    name=slot["name"],
                    arguments=arguments,
                )
                yield ToolCallEvent(call=call)
                result = tool_executor(call)
                yield ToolResultEvent(result=result)
                content = result.for_model()
                if result.images and images_refused:
                    content = f"{content}\n{_NO_IMAGES_NOTE}"
                elif result.images:
                    shown.extend(_image_parts(result))
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": content,
                    }
                )
            # Pictures follow the step's tool results in a message of their
            # own: a tool message carries text only on most servers.
            if shown:
                conversation.append({"role": "user", "content": shown})

        # The step limit ends the turn; its tool calls and results are kept.
        yield FinishEvent(
            reason="tool_limit",
            provenance=self._provenance(step.reported_model, step.response_id),
        )

    def _request(self, kwargs: Dict[str, object], interrupt: TurnInterrupt):
        """Send one request and yield its text as it streams; return the step."""
        interrupt.raise_if_set()
        stream = self.client.chat.completions.create(**cast(Any, kwargs))
        step = _Step()
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        finish_reason: Optional[str] = None
        with interrupt.watching(stream):
            for chunk in stream:
                step.reported_model = str(getattr(chunk, "model", None) or step.reported_model)
                step.response_id = str(getattr(chunk, "id", None) or step.response_id)
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    for name in _REASONING_FIELDS:
                        piece = getattr(delta, name, None)
                        if isinstance(piece, str) and piece:
                            # Servers that send both fields repeat the text.
                            step.reasoning_field = step.reasoning_field or name
                            if name == step.reasoning_field:
                                reasoning_parts.append(piece)
                    content = getattr(delta, "content", None)
                    if content:
                        text_parts.append(str(content))
                        yield TextDeltaEvent(delta=str(content))
                    for tool_delta in getattr(delta, "tool_calls", None) or []:
                        index = int(getattr(tool_delta, "index", 0) or 0)
                        slot = step.calls.setdefault(index, {"id": "", "name": "", "args": ""})
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
        if finish_reason is None:
            raise StreamCutOffError("The response stream ended before the reply was complete.")
        step.text = "".join(text_parts)
        step.reasoning = "".join(reasoning_parts)
        step.finish_reason = finish_reason
        return step


@dataclass
class _Step:
    """What one model request produced."""

    text: str = ""
    reasoning_field: str = ""
    reasoning: str = ""
    calls: Dict[int, Dict[str, str]] = field(default_factory=dict)
    finish_reason: str = "stop"
    reported_model: str = ""
    response_id: str = ""


_NO_IMAGES_NOTE = (
    "(The server refused a request that had an image in it, so images are left out "
    "for the rest of this turn.)"
)


def _image_parts(result: ToolResult) -> List[Dict[str, object]]:
    """A tool result's pictures as message parts, each after a line naming
    the call."""
    parts: List[Dict[str, object]] = []
    for image in result.images:
        parts.append({"type": "text", "text": result.image_label()})
        parts.append({"type": "image_url", "image_url": {"url": image.data_url()}})
    return parts


def _shows_images(message: Dict[str, object]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") == "image_url" for part in content
    )


def _without_images(conversation: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """The conversation as a server that takes no images is sent it: each
    message of pictures is dropped, and the tool results before it say so.
    No message is left after a step's tool results that some servers refuse."""
    kept: List[Dict[str, object]] = []
    for message in conversation:
        if not _shows_images(message):
            kept.append(message)
            continue
        if kept and kept[-1].get("role") == "tool":
            kept[-1] = {**kept[-1], "content": f"{kept[-1]['content']}\n{_NO_IMAGES_NOTE}"}
    return kept
