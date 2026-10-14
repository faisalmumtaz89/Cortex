"""Line handling of the file tools, checked against simple reference versions
on many random files."""

import io
import random
import re

import pytest

from cortex.tooling.base import ToolContext, ToolExecutionError
from cortex.tooling.builtin import read_file
from cortex.tooling.builtin.common import view_of
from cortex.tooling.builtin.edit_file import EditFileTool


def _reference_lines(data: bytes, limit: int) -> list:
    data = data.removeprefix(b"\xef\xbb\xbf")
    parts = re.split(rb"\r\n|\r|\n", data)
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    return [(part[:limit], len(part) > limit) for part in parts]


@pytest.mark.parametrize("limit", [5, 8000])
def test_read_file_splits_lines_like_the_reference_at_any_read_size(monkeypatch, limit):
    monkeypatch.setattr(read_file, "_MAX_LINE_BYTES", limit)
    pieces = [b"a", b"\r", b"\n", b"\r\n", b"abcdefgh"]
    rng = random.Random(3)
    for _ in range(3000):
        monkeypatch.setattr(read_file, "_CHUNK", rng.choice([1, 2, 3, 5, 7, 64]))
        bom = b"\xef\xbb\xbf" if rng.random() < 0.3 else b""
        data = bom + b"".join(rng.choice(pieces) for _ in range(rng.randint(0, 30)))
        want = _reference_lines(data, limit)
        skip = rng.randint(0, len(want) + 2)

        lines = read_file._Lines(io.BytesIO(data))
        passed = lines.skip(skip)
        got = []
        while (line := lines.next()) is not None:
            got.append(line)

        assert passed == min(skip, len(want)), data
        assert got == want[passed:], data


def _edit(tmp_path, data: bytes, old: str, new: str, replace_all: bool = False):
    (tmp_path / "f").write_bytes(data)
    context = ToolContext(repo_root=tmp_path.resolve(), session_id="s", call_id="c")
    arguments = {"path": "f", "old_text": old, "new_text": new, "replace_all": replace_all}
    EditFileTool().execute(arguments=arguments, context=context)
    return (tmp_path / "f").read_bytes()


def test_edits_change_exactly_what_they_say_as_read_file_shows_the_file(tmp_path):
    rng = random.Random(1)
    refused = 0
    for _ in range(3000):
        parts = [rng.choice(["a", "b", "é", ""]) for _ in range(6)]
        breaks = [rng.choice(["\r\n", "\r", "\n"]) for _ in range(5)]
        text = "".join(part + brk for part, brk in zip(parts, breaks)) + parts[-1]
        view = view_of(text)
        if not view:
            continue
        first = rng.randrange(len(view))
        old = view[first : rng.randrange(first, len(view)) + 1]
        new = rng.choice(["X", "", "\n", "X\nY", "\n\n"])
        if old == new or not old.strip("\n"):
            continue
        try:
            updated = _edit(tmp_path, text.encode(), old, new, replace_all=True)
        except ToolExecutionError:
            refused += 1
            continue
        assert view_of(updated.decode()) == view.replace(old, new), (text, old, new)
    assert refused < 300


def test_a_carriage_return_in_the_edit_text_is_not_a_line_break(tmp_path):
    with pytest.raises(ToolExecutionError, match="no text besides line breaks"):
        _edit(tmp_path, b"a\r\nb\r\n", "\r", "", replace_all=True)
    with pytest.raises(ToolExecutionError, match="differ only in line breaks"):
        _edit(tmp_path, b"a\r\nfoo\r\nb\r\n", "foo\r", "foo")


def test_new_line_breaks_do_not_join_a_neighbouring_one(tmp_path):
    assert _edit(tmp_path, b"a\rb\nc\n", "a\nb\nc", "A\n\nC") == b"A\r\n\r\nC\n"
    assert _edit(tmp_path, b"a\rb\rc\rd\ne\n", "d", "d\n") == b"a\rb\rc\rd\r\n\ne\n"
