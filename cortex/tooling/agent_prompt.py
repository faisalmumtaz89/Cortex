"""Coding-agent system prompt assembly.

One place defines what Cortex is and how it should behave. Every model
receives it as a system message alongside native tool schemas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from cortex.tooling.skills import Skill, skills_prompt

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
- Change only what the request needs, plus what the project's own rules \
(AGENTS.md, contributing docs) require for such a change. Leave other files \
alone, including existing tests, build and release tooling, and docs; do not \
add unrequested tests, refactors or fixes. Mention anything else worth doing \
in your reply instead.
- When a task leaves details unspecified (account names, passwords, ports, \
paths), use the standard convention for that tool or service.
- When asked to count or measure something, compute it with a command or \
script over the actual files or output, not by reading and counting by eye.
- Treat time as scarce: take the most direct path, start long-running \
operations (builds, boots, downloads) early and in the background, and poll \
instead of blocking.
- Reference code as path:line so the user can jump to it.
- Be concise. Lead with the answer or the outcome, not your process.
- If the user rejects a tool call, respect that and continue with what you have.\
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


def build_system_prompt(
    *,
    cwd: Path,
    skills: Optional[Dict[str, Skill]] = None,
) -> str:
    """Assemble the full system prompt for one agent turn."""
    sections = [AGENT_SYSTEM_PROMPT.format(cwd=cwd)]

    if skills:
        sections.append(skills_prompt(skills))

    project_context = load_project_context(cwd)
    if project_context:
        sections.append(project_context)

    return "\n\n".join(sections)
