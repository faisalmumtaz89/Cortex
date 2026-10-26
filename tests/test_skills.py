import os
from pathlib import Path

import pytest

from cortex.tooling.base import ToolContext, ToolExecutionError
from cortex.tooling.builtin.skill import SkillTool
from cortex.tooling.skills import load_skills


@pytest.fixture()
def home(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("cortex.tooling.skills.BUILTIN_SKILLS", tmp_path / "builtin")
    return tmp_path / "home"


def _skill(root: Path, name: str, text: str) -> Path:
    folder = root / ".cortex" / "skills" / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(text, encoding="utf-8")
    return folder


GOOD = "---\nname: good\ndescription: Use for good things.\n---\nBody.\n"


def test_a_skill_that_cannot_be_parsed_is_skipped_and_the_rest_load(tmp_path, home) -> None:
    _skill(tmp_path, "good", GOOD)
    _skill(tmp_path, "deep", "---\nname: deep\ndescription: x\nx: " + "[" * 5000 + "]" * 5000
           + "\n---\nBody.\n")
    _skill(tmp_path, "listy", "---\nname: listy\ndescription: [a, b]\n---\nBody.\n")
    _skill(tmp_path, "intname", "---\nname: 7\ndescription: x\n---\nBody.\n")
    _skill(tmp_path, "laughs", "---\nname: laughs\ndescription: &a [x, x, x, x, x, x, x, x, x]\n"
           + "".join(f"l{i}: &a{i} [*{'a' if i == 0 else f'a{i-1}'}, *{'a' if i == 0 else f'a{i-1}'}]\n"
                     for i in range(40)) + "---\nBody.\n")

    assert list(load_skills(tmp_path)) == ["good"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_unreadable_skills_and_folders_are_skipped(tmp_path, home) -> None:
    _skill(tmp_path, "good", GOOD)
    locked = _skill(tmp_path, "locked", GOOD.replace("good", "locked"))
    (locked / "SKILL.md").chmod(0)
    user = home / ".cortex" / "skills"
    user.mkdir(parents=True)
    user.chmod(0)
    try:
        assert list(load_skills(tmp_path)) == ["good"]
    finally:
        user.chmod(0o755)
        (locked / "SKILL.md").chmod(0o644)


def test_a_skill_md_with_a_byte_order_mark_loads(tmp_path, home) -> None:
    (_skill(tmp_path, "good", GOOD) / "SKILL.md").write_text("﻿" + GOOD, encoding="utf-8")

    assert list(load_skills(tmp_path)) == ["good"]


def test_files_outside_the_folder_are_not_listed_and_big_files_are_cut(tmp_path, home) -> None:
    folder = _skill(tmp_path, "good", GOOD)
    (tmp_path / "secret.txt").write_text("SECRET")
    (folder / "out").symlink_to(tmp_path / "secret.txt")
    (folder / "big.md").write_bytes(b"a" * 300_000)
    skill = load_skills(tmp_path)["good"]
    tool = SkillTool({"good": skill})
    context = ToolContext(repo_root=tmp_path, session_id="s", call_id="c")

    assert skill.files() == ["big.md"]
    with pytest.raises(ToolExecutionError, match="outside the skill"):
        tool.execute(arguments={"name": "good", "file": "out"}, context=context)
    output = tool.execute(arguments={"name": "good", "file": "big.md"}, context=context).output
    assert len(output) < 101_000 and output.endswith("[cut at 100000 bytes]")


def test_a_skill_names_the_call_that_reads_its_files(tmp_path, home) -> None:
    """A skill's files live where Cortex is installed, outside the project,
    so read_file cannot open them: the skill must name the skill-tool call
    that does, and no absolute path a model would hand to read_file."""
    folder = _skill(tmp_path, "good", GOOD)
    (folder / "examples").mkdir()
    (folder / "examples" / "plan.md").write_text("Example.", encoding="utf-8")
    skill = load_skills(tmp_path)["good"]
    tool = SkillTool({"good": skill})
    context = ToolContext(repo_root=tmp_path, session_id="s", call_id="c")

    output = tool.execute(arguments={"name": "good"}, context=context).output

    assert 'name "good"' in output and "file" in output and "- examples/plan.md" in output
    assert "read_file" in output
    assert str(folder) not in output
