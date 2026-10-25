import re
from pathlib import Path

import pytest

from cortex.tooling.registry import ToolRegistry
from cortex.tooling.skills import BUILTIN_SKILLS, load_skills

SHIPPED = ["commit-and-pr", "map-and-plan", "reproduce-and-fix", "review-changes", "safe-refactor"]


@pytest.fixture()
def shipped(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return load_skills(tmp_path)


def test_every_shipped_skill_loads(shipped) -> None:
    assert sorted(shipped) == SHIPPED
    assert sorted(p.name for p in BUILTIN_SKILLS.iterdir() if p.is_dir()) == SHIPPED


def test_shipped_skills_name_only_real_tools_and_files(shipped, tmp_path: Path) -> None:
    tools = set(ToolRegistry(repo_root=tmp_path, profile="full").names())
    for skill in shipped.values():
        assert set(skill.allowed_tools or ()) <= tools, skill.name
        text = skill.instructions()
        for name in skill.files():
            assert name in text, f"{skill.name} never points the model at {name}"
        for cited in re.findall(r"`(examples/[\w.-]+)`", text):
            assert cited in skill.files(), f"{skill.name} cites missing {cited}"


def test_read_only_skills_cannot_edit(shipped) -> None:
    assert shipped["map-and-plan"].allowed_tools == ("read_file", "list_dir", "search")
    for name in ("review-changes", "commit-and-pr"):
        allowed = set(shipped[name].allowed_tools or ())
        # They may write the file the user asks for (a review, a PR description),
        # but never edit existing files.
        assert allowed and "edit_file" not in allowed, name
    for name in ("reproduce-and-fix", "safe-refactor"):
        assert shipped[name].allowed_tools is None, name
