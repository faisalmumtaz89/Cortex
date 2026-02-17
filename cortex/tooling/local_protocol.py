"""Local-model tool protocol parsing (<tool_calls> blocks)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from cortex.tooling.types import ToolCall

TOOL_CALLS_START = "<tool_calls>"
TOOL_CALLS_END = "</tool_calls>"
TOOL_RESULTS_START = "<tool_results>"
TOOL_RESULTS_END = "</tool_results>"


def find_tool_calls_block(text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Return (start, end, inner_json) for tool_calls block, if present."""
    start = text.find(TOOL_CALLS_START)
    if start == -1:
        return None, None, None

    end = text.find(TOOL_CALLS_END, start + len(TOOL_CALLS_START))
    if end == -1:
        return start, None, None

    block = text[start + len(TOOL_CALLS_START) : end].strip()
    return start, end + len(TOOL_CALLS_END), block


def strip_tool_blocks(text: str) -> str:
    """Strip tool_calls block from text, handling incomplete blocks safely."""
    start, end, _ = find_tool_calls_block(text)
    if start is None:
        return text
    if end is None:
        return text[:start]
    return text[:start] + text[end:]


def parse_tool_calls(text: str) -> Tuple[List[ToolCall], Optional[str]]:
    """Parse tool calls from a model response.

    Returns (calls, error). When parsing fails, calls will be empty and error set.
    """
    start, end, block = find_tool_calls_block(text)
    if start is None:
        return [], None
    if end is None or block is None:
        return [], "tool_calls block is incomplete"

    try:
        payload = json.loads(block)
    except json.JSONDecodeError as exc:
        return [], f"invalid tool_calls JSON: {exc}"

    if not isinstance(payload, dict):
        return [], "tool_calls payload must be an object"

    calls_raw = payload.get("calls")
    if not isinstance(calls_raw, list):
        return [], "tool_calls payload missing 'calls' list"

    calls: List[ToolCall] = []
    for idx, call_raw in enumerate(calls_raw):
        if not isinstance(call_raw, dict):
            return [], f"tool call at index {idx} must be an object"

        name = call_raw.get("name")
        if not isinstance(name, str) or not name.strip():
            return [], f"tool call at index {idx} missing valid name"

        call_id = call_raw.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"call_{idx + 1}"

        args = call_raw.get("arguments", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return [], f"tool call '{name}' arguments must be an object"

        calls.append(ToolCall(id=call_id, name=name, arguments=args))

    return calls, None


def format_tool_results(results: List[Dict[str, Any]]) -> str:
    """Format tool results block for local-model loop continuation."""
    body = json.dumps({"results": results}, ensure_ascii=True)
    return f"{TOOL_RESULTS_START}\n{body}\n{TOOL_RESULTS_END}"
