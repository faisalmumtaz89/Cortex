"""Unit tests for the structured line-diff used by edit/write tool results."""

from __future__ import annotations

from cortex.tooling.builtin.diff_util import build_line_diff


def _rows(diff: dict) -> list[dict]:
    return [row for hunk in diff["hunks"] for row in hunk["rows"]]


def test_replace_produces_del_then_add_with_absolute_line_numbers() -> None:
    old = "def add(a, b):\n    return a + b\n"
    new = "def add(a, b):\n    # sum\n    return a + b\n"
    diff = build_line_diff(old, new, path="calc.py", op="edit", language="py")
    assert diff is not None
    assert diff["op"] == "edit"
    assert diff["language"] == "py"
    assert diff["added"] == 1
    assert diff["removed"] == 0
    kinds = [r["kind"] for r in _rows(diff)]
    assert "add" in kinds
    add_row = next(r for r in _rows(diff) if r["kind"] == "add")
    assert add_row["newNo"] == 2
    assert add_row["text"] == "    # sum"


def test_within_a_replace_all_del_rows_precede_add_rows() -> None:
    diff = build_line_diff("a = 1\n", "a = 2\n", path="x.py", op="edit")
    assert diff is not None
    kinds = [r["kind"] for r in _rows(diff)]
    assert kinds == ["del", "add"]
    assert _rows(diff)[0]["oldNo"] == 1  # del carries old number
    assert _rows(diff)[1]["newNo"] == 1  # add carries new number


def test_new_file_is_one_all_add_hunk() -> None:
    diff = build_line_diff("", "one\ntwo\nthree\n", path="new.py", op="create")
    assert diff is not None
    assert diff["op"] == "create"
    assert diff["added"] == 3
    assert diff["removed"] == 0
    assert all(r["kind"] == "add" for r in _rows(diff))
    assert [r["newNo"] for r in _rows(diff)] == [1, 2, 3]


def test_identical_content_returns_none() -> None:
    assert build_line_diff("same\ncontent\n", "same\ncontent\n", path="x", op="edit") is None


def test_empty_create_returns_none() -> None:
    assert build_line_diff("", "", path="x", op="create") is None


def test_multi_hunk_changes_split_into_separate_hunks() -> None:
    old = "\n".join(f"line{i}" for i in range(1, 21)) + "\n"
    lines = [f"line{i}" for i in range(1, 21)]
    lines[1] = "CHANGED2"
    lines[18] = "CHANGED19"
    new = "\n".join(lines) + "\n"
    diff = build_line_diff(old, new, path="x.txt", op="edit", context=3)
    assert diff is not None
    assert len(diff["hunks"]) == 2  # far-apart changes are separate hunks
    # Second hunk's numbers are absolute (not restarted at 1).
    second = diff["hunks"][1]
    assert second["rows"][0]["newNo"] > 10


def test_row_cap_truncates_and_reports_true_pre_truncation_counts() -> None:
    old = "".join(f"{i}\n" for i in range(600))
    new = "".join(f"x{i}\n" for i in range(600))
    diff = build_line_diff(old, new, path="big.txt", op="edit", max_rows=50)
    assert diff is not None
    assert diff["truncated"] is True
    assert diff["truncatedRows"] > 0
    total_rows = sum(len(h["rows"]) for h in diff["hunks"])
    assert total_rows <= 50
    # Counts stay exact (pre-truncation): 600 removed + 600 added.
    assert diff["added"] == 600
    assert diff["removed"] == 600


def test_huge_file_short_circuits_to_none() -> None:
    big = "\n".join(str(i) for i in range(20001))
    assert build_line_diff(big, big + "\nextra", path="x", op="edit") is None


def test_long_line_is_clipped_with_flag() -> None:
    diff = build_line_diff("short\n", "y" * 3000 + "\n", path="x", op="edit", max_row_chars=100)
    assert diff is not None
    add_row = next(r for r in _rows(diff) if r["kind"] == "add")
    assert add_row.get("clipped") is True
    assert len(add_row["text"]) <= 101  # 100 + the ellipsis


def test_crlf_is_normalized_not_counted_as_changes() -> None:
    diff = build_line_diff("a\r\nb\r\n", "a\nb\n", path="x", op="edit")
    assert diff is None  # only line endings differ → no visible change
