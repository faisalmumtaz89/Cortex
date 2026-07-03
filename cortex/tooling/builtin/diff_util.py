"""Structured line-diff computation for edit/write tool results.

Produces a compact, UI-ready diff (hunks of added/removed/context rows with
line numbers) placed in ToolResult.metadata["diff"]; the TUI renders it as a
green/red diff. Dependency-free (stdlib difflib only, so it survives the
compiled single-binary sidecar).
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional

MAX_DIFF_ROWS = 400  # hard cap on rows across all hunks (bounds the wire payload)
MAX_ROW_CHARS = 2000  # per-line char cap (one minified line can't bloat metadata)
MAX_LINES = 20000  # skip the O(n*m) matcher above this on either side


def _split_lines(text: str) -> List[str]:
    # Normalize line endings ourselves (splitlines() over-splits on \x0b/\x0c/
    # U+2028…), then drop the single empty element a terminal newline produces.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _row(kind: str, text: str, max_row_chars: int, **numbers: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {"kind": kind}
    if len(text) > max_row_chars:
        row["text"] = text[:max_row_chars] + "…"
        row["clipped"] = True
    else:
        row["text"] = text
    row.update(numbers)
    return row


def build_line_diff(
    old_text: str,
    new_text: str,
    *,
    path: str,
    op: str,
    language: Optional[str] = None,
    context: int = 3,
    max_rows: int = MAX_DIFF_ROWS,
    max_row_chars: int = MAX_ROW_CHARS,
    max_lines: int = MAX_LINES,
) -> Optional[Dict[str, Any]]:
    """Build the metadata.diff object, or None when there is nothing to show
    (identical content, or a file too large to diff inline — the caller then
    falls back to its plain one-line summary)."""
    old_lines = _split_lines(old_text)
    new_lines = _split_lines(new_text)

    if len(old_lines) > max_lines or len(new_lines) > max_lines:
        return None  # too large to diff inline; plain tool summary stands in

    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    groups = list(matcher.get_grouped_opcodes(context))
    if not groups:
        return None  # identical

    added = 0
    removed = 0
    hunks: List[Dict[str, Any]] = []
    for group in groups:
        old_start = group[0][1] + 1
        new_start = group[0][3] + 1
        rows: List[Dict[str, Any]] = []
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for offset in range(i2 - i1):
                    rows.append(
                        _row(
                            "ctx",
                            old_lines[i1 + offset],
                            max_row_chars,
                            oldNo=i1 + offset + 1,
                            newNo=j1 + offset + 1,
                        )
                    )
                continue
            if tag in ("delete", "replace"):
                for offset in range(i2 - i1):
                    removed += 1
                    rows.append(_row("del", old_lines[i1 + offset], max_row_chars, oldNo=i1 + offset + 1))
            if tag in ("insert", "replace"):
                for offset in range(j2 - j1):
                    added += 1
                    rows.append(_row("add", new_lines[j1 + offset], max_row_chars, newNo=j1 + offset + 1))
        hunks.append({"oldStart": old_start, "newStart": new_start, "rows": rows})

    # Row cap: keep whole hunks in order; clip the overflowing hunk; drop the rest.
    capped: List[Dict[str, Any]] = []
    truncated = False
    truncated_rows = 0
    budget = max_rows
    for hunk in hunks:
        row_count = len(hunk["rows"])
        if budget <= 0:
            truncated = True
            truncated_rows += row_count
            continue
        if row_count <= budget:
            capped.append(hunk)
            budget -= row_count
        else:
            truncated = True
            truncated_rows += row_count - budget
            capped.append({**hunk, "rows": hunk["rows"][:budget]})
            budget = 0

    result: Dict[str, Any] = {
        "path": path,
        "op": op,
        "added": added,
        "removed": removed,
        "truncated": truncated,
        "truncatedRows": truncated_rows,
        "hunks": capped,
    }
    if language:
        result["language"] = language
    return result
