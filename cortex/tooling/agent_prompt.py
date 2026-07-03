"""Coding-agent system prompt assembly.

One place defines what Cortex is and how it should behave. The same prompt is
used for cloud models (as a system message alongside native tool schemas) and
local models (prepended to the chat template, with the <tool_calls> protocol
appended because local models have no native tool-calling).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from cortex.tooling.types import ToolSpec

# Project context files, in priority order. AGENTS.md is the cross-tool standard.
PROJECT_CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md")
_MAX_PROJECT_CONTEXT_BYTES = 16_384

AGENT_SYSTEM_PROMPT = """\
You are Cortex, an AI coding agent running in the user's terminal.
Working directory: {cwd}

You help with software engineering tasks: navigating and explaining code, \
fixing bugs, writing and refactoring code, and running commands.

Rules:
- Ground every claim about this repository in tool output. Read the actual \
files before describing or changing them; never guess paths, APIs, or contents.
- Search first (search, list_dir), then read the smallest useful region \
(read_file with line ranges).
- Prefer edit_file for surgical changes; old_text must match the file exactly. \
Use write_file only for new files or full rewrites.
- After changing code, verify it: re-read the edited region or run the \
project's own checks (tests, build, linters) with bash.
- The end state is the deliverable. When done, delete scratch files and build \
artifacts you created while testing, keep any requested services running, and \
never undo or reset the work the task asked for.
- When a task leaves details unspecified (account names, passwords, ports, \
paths), use the standard convention for that tool or service — and when it is \
cheap, satisfy the other plausible readings too rather than betting on one.
- Treat time as scarce: take the most direct path, start long-running \
operations (builds, boots, downloads) early and in the background, and poll \
instead of blocking.
- Reference code as path:line so the user can jump to it.
- Be concise. Lead with the answer or the outcome, not your process.
- If the user rejects a tool call, respect that and continue with what you have.\
"""

LOCAL_TOOL_PROTOCOL = """\
Tool calling protocol:
You may call the tools listed below. To call tools, end your reply with exactly \
one block in this form (valid JSON, double quotes):
<tool_calls>{"calls": [{"name": "<tool_name>", "arguments": {...}}]}</tool_calls>
Results arrive in a <tool_results> JSON block in a system message. Keep calling \
tools until you can answer, then reply WITHOUT any <tool_calls> block.

Available tools:
"""


def load_project_context(root: Path) -> Optional[str]:
    """Return the project's agent instructions (AGENTS.md), if present."""
    for name in PROJECT_CONTEXT_FILES:
        candidate = root / name
        try:
            if not candidate.is_file():
                continue
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        if len(text.encode("utf-8")) > _MAX_PROJECT_CONTEXT_BYTES:
            text = text.encode("utf-8")[:_MAX_PROJECT_CONTEXT_BYTES].decode(
                "utf-8", errors="ignore"
            )
            text += "\n[project context truncated]"
        return f"Project instructions from {name}:\n{text}"
    return None


def _describe_tools(specs: List[ToolSpec]) -> str:
    lines = []
    for spec in specs:
        params = json.dumps(spec.parameters.get("properties", {}), ensure_ascii=True)
        lines.append(f"- {spec.name}: {spec.description} Parameters: {params}")
    return "\n".join(lines)


def build_system_prompt(
    *,
    cwd: Path,
    tool_specs: Optional[List[ToolSpec]] = None,
    include_local_protocol: bool = False,
) -> str:
    """Assemble the full system prompt for one agent turn."""
    sections = [AGENT_SYSTEM_PROMPT.format(cwd=cwd)]

    project_context = load_project_context(cwd)
    if project_context:
        sections.append(project_context)

    if include_local_protocol and tool_specs:
        sections.append(LOCAL_TOOL_PROTOCOL + _describe_tools(tool_specs))

    return "\n\n".join(sections)
