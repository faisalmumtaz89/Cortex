"""Unit tests for the edit_file and write_file tools and agent prompt assembly."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.tooling.agent_prompt import build_system_prompt, load_project_context
from cortex.tooling.base import ToolContext, ToolExecutionError
from cortex.tooling.builtin import EditFileTool, WriteFileTool
from cortex.tooling.registry import ToolRegistry


def _context(root: Path) -> ToolContext:
    return ToolContext(repo_root=root.resolve(), session_id="s1", call_id="c1")


def test_edit_file_replaces_unique_text(tmp_path: Path) -> None:
    target = tmp_path / "example.py"
    target.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")

    result = EditFileTool().execute(
        arguments={"path": "example.py", "old_text": "'hello'", "new_text": "'world'"},
        context=_context(tmp_path),
    )

    assert result.ok is True
    assert target.read_text(encoding="utf-8") == "def greet():\n    return 'world'\n"
    assert result.metadata["replacements"] == 1


def test_edit_file_rejects_ambiguous_match(tmp_path: Path) -> None:
    target = tmp_path / "example.py"
    target.write_text("x = 1\nx = 1\n", encoding="utf-8")

    with pytest.raises(ToolExecutionError, match="2 locations"):
        EditFileTool().execute(
            arguments={"path": "example.py", "old_text": "x = 1", "new_text": "x = 2"},
            context=_context(tmp_path),
        )


def test_edit_file_replace_all(tmp_path: Path) -> None:
    target = tmp_path / "example.py"
    target.write_text("x = 1\nx = 1\n", encoding="utf-8")

    result = EditFileTool().execute(
        arguments={
            "path": "example.py",
            "old_text": "x = 1",
            "new_text": "x = 2",
            "replace_all": True,
        },
        context=_context(tmp_path),
    )

    assert result.metadata["replacements"] == 2
    assert target.read_text(encoding="utf-8") == "x = 2\nx = 2\n"


def test_edit_file_rejects_missing_text(tmp_path: Path) -> None:
    (tmp_path / "example.py").write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(ToolExecutionError, match="not found"):
        EditFileTool().execute(
            arguments={"path": "example.py", "old_text": "y = 2", "new_text": "y = 3"},
            context=_context(tmp_path),
        )


def test_edit_file_rejects_escaping_path(tmp_path: Path) -> None:
    with pytest.raises(ToolExecutionError, match="outside the working directory"):
        EditFileTool().execute(
            arguments={"path": "../outside.py", "old_text": "a", "new_text": "b"},
            context=_context(tmp_path),
        )


def test_write_file_creates_nested_file(tmp_path: Path) -> None:
    result = WriteFileTool().execute(
        arguments={"path": "pkg/module.py", "content": "VALUE = 42\n"},
        context=_context(tmp_path),
    )

    assert result.ok is True
    assert result.metadata["created"] is True
    assert (tmp_path / "pkg" / "module.py").read_text(encoding="utf-8") == "VALUE = 42\n"


def test_write_file_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("old", encoding="utf-8")

    result = WriteFileTool().execute(
        arguments={"path": "notes.txt", "content": "new"},
        context=_context(tmp_path),
    )

    assert result.metadata["created"] is False
    assert target.read_text(encoding="utf-8") == "new"


def test_write_file_refuses_binary_overwrite(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")

    with pytest.raises(ToolExecutionError, match="binary"):
        WriteFileTool().execute(
            arguments={"path": "blob.bin", "content": "text"},
            context=_context(tmp_path),
        )


def test_registry_profiles_gate_tools(tmp_path: Path) -> None:
    read_only = ToolRegistry(repo_root=tmp_path, profile="read_only")
    edit = ToolRegistry(repo_root=tmp_path, profile="edit")
    full = ToolRegistry(repo_root=tmp_path, profile="full")

    assert read_only.names() == ["list_dir", "read_file", "search"]
    assert edit.names() == ["edit_file", "list_dir", "read_file", "search", "write_file"]
    assert full.names() == ["bash", "edit_file", "list_dir", "read_file", "search", "write_file"]


def test_project_context_prefers_agents_md(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Run make test before committing.", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("ignored", encoding="utf-8")

    context = load_project_context(tmp_path)

    assert context is not None
    assert "AGENTS.md" in context
    assert "make test" in context


def test_system_prompt_has_no_local_tool_protocol(tmp_path: Path) -> None:
    # Local models call tools natively through Lumen's OpenAI-compatible
    # server; the prompt must not carry the old <tool_calls> text protocol.
    prompt = build_system_prompt(cwd=tmp_path)

    assert "AI coding agent" in prompt
    assert "<tool_calls>" not in prompt


def test_system_prompt_without_project_context(tmp_path: Path) -> None:
    prompt = build_system_prompt(cwd=tmp_path)

    assert "Project instructions" not in prompt
    assert str(tmp_path) in prompt
