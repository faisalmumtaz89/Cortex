"""Skills: instructions for a kind of task, loaded when a request needs them.

A skill is a folder holding SKILL.md: YAML front matter with its `name`, a
`description` that says when it applies, and optionally `allowed-tools`, the
only tools it uses; then the instructions. Other files in the folder
(examples, references) are read on demand. The system prompt lists each
skill's name and description; the `skill` tool loads the rest.

Skills come from three places, the more specific overriding the less: the
skills shipped with Cortex, the user's ~/.cortex/skills, and the project's
.cortex/skills, which Cortex trusts as it trusts the project's AGENTS.md.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

BUILTIN_SKILLS = Path(__file__).resolve().parent.parent / "skills"
_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MAX_DESCRIPTION = 400
_MAX_SKILL_MD_BYTES = 1_000_000


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    root: Path
    allowed_tools: Optional[Tuple[str, ...]]  # None: any tool

    def instructions(self) -> str:
        return _split(self.root / "SKILL.md")[1].strip()

    def files(self) -> List[str]:
        """The skill's other files, relative to its folder."""
        found = []
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if relative.as_posix() == "SKILL.md" or not path.is_file():
                continue
            # The skill tool reads only files inside the folder; list only those.
            if path.resolve().is_relative_to(self.root):
                found.append(relative.as_posix())
        return sorted(found)


def skill_dirs(repo_root: Path) -> List[Path]:
    """Where skills are found, least specific first."""
    return [BUILTIN_SKILLS, Path.home() / ".cortex" / "skills", repo_root / ".cortex" / "skills"]


def load_skills(repo_root: Path) -> Dict[str, Skill]:
    """Every valid skill by name; a skill in a more specific place replaces
    one of the same name. Invalid skills are skipped and logged."""
    skills: Dict[str, Skill] = {}
    for base in skill_dirs(repo_root):
        try:
            folders = sorted(base.iterdir()) if base.is_dir() else []
        except OSError as exc:
            logger.warning("Could not list the skills in %s: %s", base, exc)
            continue
        for folder in folders:
            try:
                if not (folder / "SKILL.md").is_file():
                    continue
                skill = _read(folder)
            except Exception as exc:  # one bad skill must not stop the others or the turn
                logger.warning("Skipped the skill in %s: %s", folder, exc)
                continue
            skills[skill.name] = skill
    return skills


def _read(folder: Path) -> Skill:
    header, _ = _split(folder / "SKILL.md")
    name = header.get("name")
    if not isinstance(name, str) or not _NAME.match(name) or name != folder.name:
        raise ValueError("its name must be lowercase-with-hyphens and match its folder")
    description = header.get("description")
    if not isinstance(description, str):
        raise ValueError("its description must be text")
    description = " ".join(description.split())
    if not description:
        raise ValueError("it has no description")
    if len(description) > _MAX_DESCRIPTION:
        raise ValueError(f"its description is longer than {_MAX_DESCRIPTION} characters")
    allowed = header.get("allowed-tools")
    if allowed is not None and not (
        isinstance(allowed, list) and all(isinstance(tool, str) for tool in allowed)
    ):
        raise ValueError("allowed-tools must be a list of tool names")
    return Skill(
        name=name,
        description=description,
        root=folder.resolve(),
        allowed_tools=tuple(allowed) if allowed is not None else None,
    )


def _split(path: Path) -> Tuple[dict, str]:
    """The front matter and the body of a SKILL.md."""
    if path.stat().st_size > _MAX_SKILL_MD_BYTES:
        raise ValueError(f"SKILL.md is larger than {_MAX_SKILL_MD_BYTES} bytes")
    text = path.read_text(encoding="utf-8-sig")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", text, re.DOTALL)
    if not match:
        raise ValueError("SKILL.md must start with front matter between --- lines")
    try:
        header = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"its front matter is not valid YAML: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("its front matter must be key: value pairs")
    return header, text[match.end():]


def skills_prompt(skills: Dict[str, Skill]) -> str:
    """The system prompt's list of skills."""
    lines = [
        "Skills: instructions for kinds of task. When a request matches a skill's "
        "description, call the skill tool with its name before anything else, and "
        "follow what it returns.",
    ]
    lines += [f"- {skill.name}: {skill.description}" for skill in skills.values()]
    return "\n".join(lines)
