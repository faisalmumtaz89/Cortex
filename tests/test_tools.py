import shutil
from pathlib import Path

import pytest

from cortex.tools import protocol as tool_protocol
from cortex.tools.errors import ToolError, ValidationError
from cortex.tools.fs_ops import RepoFS
from cortex.tools.search import RepoSearch
from cortex.tools.tool_runner import ToolRunner


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


def test_parse_tool_calls_valid():
    payload = "<tool_calls>{\"calls\":[{\"id\":\"call_1\",\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"}}]}</tool_calls>"
    calls, error = tool_protocol.parse_tool_calls(payload)
    assert error is None
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["name"] == "list_dir"


def test_parse_tool_calls_invalid_json():
    payload = "<tool_calls>{invalid}</tool_calls>"
    calls, error = tool_protocol.parse_tool_calls(payload)
    assert calls == []
    assert error is not None


def test_resolve_path_rejects_absolute(tmp_path: Path):
    root = _make_repo(tmp_path)
    fs = RepoFS(root)
    with pytest.raises(ValidationError):
        fs.resolve_path(str(root / "file.txt"))


def test_read_text_range(tmp_path: Path):
    root = _make_repo(tmp_path)
    (root / "sample.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    fs = RepoFS(root)
    result = fs.read_text("sample.txt", start_line=2, end_line=3)
    assert result["content"] == "beta\ngamma"
    assert result["start_line"] == 2
    assert result["end_line"] == 3


def test_read_text_invalid_utf8(tmp_path: Path):
    root = _make_repo(tmp_path)
    (root / "bad.bin").write_bytes(b"\xff\xfe\xff")
    fs = RepoFS(root)
    with pytest.raises(ToolError):
        fs.read_text("bad.bin")


def test_write_text_expected_hash_mismatch(tmp_path: Path):
    root = _make_repo(tmp_path)
    (root / "sample.txt").write_text("original", encoding="utf-8")
    fs = RepoFS(root)
    with pytest.raises(ToolError):
        fs.write_text("sample.txt", "updated", expected_sha256="0" * 64)


def test_create_text_overwrite_must_be_bool(tmp_path: Path):
    root = _make_repo(tmp_path)
    fs = RepoFS(root)
    with pytest.raises(ValidationError):
        fs.create_text("new.txt", "content", overwrite="yes")


def test_search_python_handles_file(monkeypatch, tmp_path: Path):
    root = _make_repo(tmp_path)
    (root / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")
    fs = RepoFS(root)
    search = RepoSearch(fs)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    results = search.search("world", path="notes.txt", use_regex=False, max_results=10)
    assert len(results["results"]) == 1
    assert results["results"][0]["line"] == 2


def test_search_invalid_regex(monkeypatch, tmp_path: Path):
    root = _make_repo(tmp_path)
    (root / "notes.txt").write_text("hello\n", encoding="utf-8")
    fs = RepoFS(root)
    search = RepoSearch(fs)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(ValidationError):
        search.search("(", path="notes.txt", use_regex=True, max_results=10)


def test_tool_runner_create_rejects_directory_path(tmp_path: Path):
    root = _make_repo(tmp_path)
    runner = ToolRunner(root)
    called = False

    def confirm(_prompt: str) -> bool:
        nonlocal called
        called = True
        return True

    runner.set_confirm_callback(confirm)
    results = runner.run_calls(
        [
            {
                "id": "call_1",
                "name": "create_file",
                "arguments": {"path": ".", "content": "x", "overwrite": True},
            }
        ]
    )
    assert results[0]["ok"] is False
    assert called is False


def test_tool_runner_delete_untracked_skips_prompt(tmp_path: Path):
    root = _make_repo(tmp_path)
    (root / "untracked.txt").write_text("content", encoding="utf-8")
    runner = ToolRunner(root)
    called = False

    def confirm(_prompt: str) -> bool:
        nonlocal called
        called = True
        return True

    runner.set_confirm_callback(confirm)
    results = runner.run_calls(
        [{"id": "call_1", "name": "delete_file", "arguments": {"path": "untracked.txt"}}]
    )
    assert results[0]["ok"] is False
    assert "not tracked" in (results[0]["error"] or "")
    assert called is False


def test_tool_runner_write_invalid_hash_skips_prompt(tmp_path: Path):
    root = _make_repo(tmp_path)
    (root / "sample.txt").write_text("original", encoding="utf-8")
    runner = ToolRunner(root)
    called = False

    def confirm(_prompt: str) -> bool:
        nonlocal called
        called = True
        return True

    runner.set_confirm_callback(confirm)
    results = runner.run_calls(
        [
            {
                "id": "call_1",
                "name": "write_file",
                "arguments": {"path": "sample.txt", "content": "updated", "expected_sha256": "bad"},
            }
        ]
    )
    assert results[0]["ok"] is False
    assert "expected_sha256" in (results[0]["error"] or "")
    assert called is False
