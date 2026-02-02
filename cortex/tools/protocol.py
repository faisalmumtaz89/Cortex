"""Protocol helpers for tool calling."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

TOOL_CALLS_START = "<tool_calls>"
TOOL_CALLS_END = "</tool_calls>"
TOOL_RESULTS_START = "<tool_results>"
TOOL_RESULTS_END = "</tool_results>"


def find_tool_calls_block(text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Return (start, end, block) for tool_calls JSON, if present."""
    start = text.find(TOOL_CALLS_START)
    if start == -1:
        return None, None, None
    end = text.find(TOOL_CALLS_END, start + len(TOOL_CALLS_START))
    if end == -1:
        return start, None, None
    block = text[start + len(TOOL_CALLS_START) : end].strip()
    return start, end + len(TOOL_CALLS_END), block


def strip_tool_blocks(text: str) -> str:
    """Remove tool_calls block from text (including incomplete block)."""
    start, end, _ = find_tool_calls_block(text)
    if start is None:
        return text
    if end is None:
        return text[:start]
    return text[:start] + text[end:]


def parse_tool_calls(text: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Parse tool calls from text. Returns (calls, error)."""
    start, end, block = find_tool_calls_block(text)
    if start is None:
        return [], None
    if end is None or block is None:
        return [], "tool_calls block is incomplete"
    try:
        payload = json.loads(block)
    except json.JSONDecodeError as e:
        return [], f"invalid tool_calls JSON: {e}"

    if not isinstance(payload, dict):
        return [], "tool_calls payload must be a JSON object"
    calls = payload.get("calls")
    if not isinstance(calls, list):
        return [], "tool_calls payload missing 'calls' list"

    normalized: List[Dict[str, Any]] = []
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            return [], f"tool call at index {idx} must be an object"
        name = call.get("name")
        arguments = call.get("arguments")
        call_id = call.get("id") or f"call_{idx + 1}"
        if not isinstance(name, str) or not name.strip():
            return [], f"tool call at index {idx} missing valid name"
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return [], f"tool call '{name}' arguments must be an object"
        normalized.append({"id": str(call_id), "name": name, "arguments": arguments})

    return normalized, None


def format_tool_results(results: List[Dict[str, Any]]) -> str:
    """Format tool results for model consumption."""
    payload = {"results": results}
    body = json.dumps(payload, ensure_ascii=True)
    return f"{TOOL_RESULTS_START}\n{body}\n{TOOL_RESULTS_END}"
