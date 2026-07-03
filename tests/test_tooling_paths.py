"""Path sandbox resolution rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.tooling.builtin.common import resolve_repo_path


def test_relative_path_resolves(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert resolve_repo_path(root=tmp_path, relative_path="a.txt") == (
        tmp_path / "a.txt"
    ).resolve()


def test_absolute_path_inside_root_is_accepted(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "b.txt"
    resolved = resolve_repo_path(root=tmp_path.resolve(), relative_path=str(target))
    assert resolved == target.resolve()


def test_absolute_path_outside_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the working directory"):
        resolve_repo_path(root=tmp_path.resolve(), relative_path="/etc/passwd")


def test_traversal_escape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the working directory"):
        resolve_repo_path(root=tmp_path.resolve(), relative_path="../escape.txt")
