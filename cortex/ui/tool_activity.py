"""Tool activity rendering helpers for the CLI."""

from __future__ import annotations

from typing import List

from rich.console import Console
from rich.style import Style
from rich.text import Text

from cortex.ui.markdown_render import PrefixedRenderable


def summarize_tool_call(call: dict) -> str:
    name = str(call.get("name", "tool"))
    args = call.get("arguments") or {}
    parts: List[str] = []
    preferred = ("path", "query", "anchor", "start_line", "end_line", "recursive", "max_results")
    for key in preferred:
        if key in args:
            value = args[key]
            if isinstance(value, str) and len(value) > 60:
                value = value[:57] + "..."
            parts.append(f"{key}={value!r}")
    if not parts and args:
        for key in list(args.keys())[:3]:
            value = args[key]
            if isinstance(value, str) and len(value) > 60:
                value = value[:57] + "..."
            parts.append(f"{key}={value!r}")
    arg_str = ", ".join(parts)
    return f"{name}({arg_str})" if arg_str else f"{name}()"


def summarize_tool_result(result: dict) -> str:
    name = str(result.get("name", "tool"))
    if not result.get("ok", False):
        error = result.get("error") or "unknown error"
        return f"{name} -> error: {error}"
    payload = result.get("result") or {}
    if name == "list_dir":
        entries = payload.get("entries") or []
        return f"{name} -> entries={len(entries)}"
    if name == "search":
        matches = payload.get("results") or []
        return f"{name} -> results={len(matches)}"
    if name == "read_file":
        path = payload.get("path") or ""
        start = payload.get("start_line")
        end = payload.get("end_line")
        if start and end:
            return f"{name} -> {path} lines {start}-{end}"
        if start:
            return f"{name} -> {path} from line {start}"
        return f"{name} -> {path}"
    if name in {"write_file", "create_file", "delete_file", "replace_in_file", "insert_after", "insert_before"}:
        path = payload.get("path") or ""
        return f"{name} -> {path}"
    return f"{name} -> ok"


def print_tool_activity(
    console: Console,
    tool_calls: list,
    tool_results: list,
    terminal_width: int,
) -> None:
    lines = []
    for call, result in zip(tool_calls, tool_results):
        lines.append(f"tool {summarize_tool_call(call)} -> {summarize_tool_result(result)}")
    if not lines:
        return
    text = Text("\n".join(lines), style=Style(color="bright_black", italic=True))
    renderable = PrefixedRenderable(text, prefix="  ", prefix_style=Style(dim=True), indent="  ", auto_space=False)
    original_console_width = console._width
    target_width = max(40, int(terminal_width * 0.75))
    console.width = target_width
    try:
        console.print(renderable, highlight=False, soft_wrap=True)
        console.print()
    finally:
        console._width = original_console_width
