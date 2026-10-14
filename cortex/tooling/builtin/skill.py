"""Read-only tool that loads a skill's instructions, or one of its files."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.skills import Skill
from cortex.tooling.types import ToolExecutionState, ToolResult

_MAX_BYTES = 100_000
_MAX_FILES = 100


class SkillArgs(BaseModel):
    name: str = Field(description="The skill's name, as the system prompt lists it.")
    file: Optional[str] = Field(
        default=None,
        description="One of the skill's files to read, relative to its folder. "
        "Leave it out to load the skill's instructions.",
    )


class SkillTool(BaseTool[SkillArgs]):
    name = "skill"
    description = (
        "Load a skill's instructions, before starting a task its description matches. "
        "With file, read one of the files the instructions list."
    )
    permission = "read"
    args_model = SkillArgs

    def __init__(self, skills: Dict[str, Skill]):
        self.skills = skills

    def run(self, parsed_args: SkillArgs, context: ToolContext) -> ToolResult:
        skill = self.skills.get(parsed_args.name.strip())
        if skill is None:
            raise ToolExecutionError(
                f"no skill is named {parsed_args.name!r}; the skills are: "
                + ", ".join(sorted(self.skills))
            )
        try:
            if parsed_args.file is None:
                output = skill_text(skill)
            else:
                output = self._file(skill, parsed_args.file)
        except OSError as exc:
            raise ToolExecutionError(f"could not read the {skill.name} skill: {exc}") from exc
        except ValueError as exc:
            raise ToolExecutionError(f"the {skill.name} skill is not valid: {exc}") from exc
        return ToolResult(
            id=context.call_id,
            name=self.name,
            state=ToolExecutionState.COMPLETED,
            ok=True,
            output=output,
            metadata={
                "skill": skill.name,
                "file": parsed_args.file,
                "allowed_tools": list(skill.allowed_tools)
                if skill.allowed_tools is not None
                else None,
            },
        )

    def _file(self, skill: Skill, file: str) -> str:
        path = (skill.root / file).resolve()
        if not path.is_relative_to(skill.root) or path == skill.root:
            raise ToolExecutionError(f"{file} is outside the skill's folder")
        if not path.is_file():
            raise ToolExecutionError(f"the {skill.name} skill has no file {file}")
        with path.open("rb") as handle:
            data = handle.read(_MAX_BYTES + 1)
        if b"\x00" in data[:8192]:
            raise ToolExecutionError(f"{file} is a binary file")
        return _cut(data.decode("utf-8", errors="replace"))


def skill_text(skill: Skill) -> str:
    """A skill's instructions, with its files and the tools it limits a turn to."""
    sections = [_cut(skill.instructions())]
    files = skill.files()
    if files:
        listed = "\n".join(f"- {path}" for path in files[:_MAX_FILES])
        if len(files) > _MAX_FILES:
            listed += f"\n- … and {len(files) - _MAX_FILES} more"
        sections.append(
            f"Files of this skill, in {skill.root}; read one with the skill tool "
            f"and its path as file:\n{listed}"
        )
    if skill.allowed_tools is not None:
        sections.append(
            "This skill uses only these tools: "
            + (", ".join(skill.allowed_tools) or "none")
            + ". Other tools are refused for the rest of this turn."
        )
    return "\n\n".join(sections)


def _cut(text: str) -> str:
    data = text.encode("utf-8")
    if len(data) <= _MAX_BYTES:
        return text
    kept = data[:_MAX_BYTES].decode("utf-8", errors="ignore")
    return f"{kept}\n[cut at {_MAX_BYTES} bytes]"
