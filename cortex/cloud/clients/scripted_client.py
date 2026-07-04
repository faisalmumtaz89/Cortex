"""Deterministic scripted model client for end-to-end runtime validation.

Activated by setting CORTEX_SCRIPTED_MODEL to a JSON script path. The worker
then runs its real agent loop — orchestrator, registry, permissions, events,
persistence — against replayed model output instead of a live provider.

Script format:
{
  "responses": [
    [
      {"text": "optional text", "tool_calls": [{"name": "...", "arguments": {...}}]},
      {"text": "final text for this user turn"}
    ]
  ]
}

Each entry in "responses" serves one user turn (selected by counting user
messages in the request). Each step models one model round-trip: its text is
streamed, its tool calls are executed, and the loop continues to the next
step — exactly the shape of the native cloud tool loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

from cortex.tooling.types import (
    FinishEvent,
    TextDeltaEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
)

_STREAM_CHUNK_CHARS = 24


class ScriptedClient:
    """Replays scripted model responses through the cloud client interface."""

    def __init__(self, script_path: str):
        self.script_path = Path(script_path)
        payload = json.loads(self.script_path.read_text(encoding="utf-8"))
        responses = payload.get("responses")
        if not isinstance(responses, list) or not responses:
            raise ValueError(f"scripted model file has no responses: {script_path}")
        self.responses: List[List[Dict[str, object]]] = responses

    def _select_response(self, messages: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        user_turns = sum(1 for message in messages if message.get("role") == "user")
        index = max(0, user_turns - 1)
        if index >= len(self.responses):
            raise RuntimeError(
                f"scripted model exhausted: turn {index + 1} requested, "
                f"{len(self.responses)} scripted"
            )
        steps = self.responses[index]
        if not isinstance(steps, list):
            raise ValueError(f"scripted response {index} must be a list of steps")
        return steps

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
        max_tool_iterations: int = 25,
    ):
        steps = self._select_response(list(messages))
        provenance = {
            "client_kind": "scripted",
            "reported_model": model_id,
            "response_id": "scripted",
            "endpoint": str(self.script_path),
        }

        iterations = 0
        for step in steps:
            iterations += 1
            if iterations > max_tool_iterations:
                raise RuntimeError("Tool loop limit reached for scripted response")

            # Optional pacing so UI tests can observe busy states deterministically.
            delay_ms = step.get("delay_ms")
            if isinstance(delay_ms, (int, float)) and delay_ms > 0:
                time.sleep(float(delay_ms) / 1000.0)

            text = str(step.get("text", "") or "")
            for offset in range(0, len(text), _STREAM_CHUNK_CHARS):
                yield TextDeltaEvent(delta=text[offset : offset + _STREAM_CHUNK_CHARS])

            raw_calls = step.get("tool_calls")
            if not isinstance(raw_calls, list) or not raw_calls:
                yield FinishEvent(reason="stop", provenance=provenance)
                return

            if tool_executor is None:
                raise RuntimeError("scripted response contains tool_calls but no executor is set")

            for call_index, raw_call in enumerate(raw_calls):
                call = ToolCall(
                    id=str(raw_call.get("id") or f"scripted_{iterations}_{call_index + 1}"),
                    name=str(raw_call.get("name", "")),
                    arguments=dict(raw_call.get("arguments") or {}),
                )
                yield ToolCallEvent(call=call)
                result: ToolResult = tool_executor(call)
                yield ToolResultEvent(result=result)

        yield FinishEvent(reason="stop", provenance=provenance)
