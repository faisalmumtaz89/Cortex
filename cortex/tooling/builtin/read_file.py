"""Read-only file read tool."""

from __future__ import annotations

import codecs
import re
from typing import BinaryIO, Optional, Tuple

from pydantic import BaseModel, Field

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.builtin.common import is_binary_file, resolve_repo_path
from cortex.tooling.types import ToolExecutionState, ToolResult

# One call shows at most this many lines, characters per line, and bytes.
_MAX_LINES = 2000
_MAX_LINE_CHARS = 2000
_MAX_OUTPUT_BYTES = 100_000


class ReadFileArgs(BaseModel):
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class ReadFileTool(BaseTool[ReadFileArgs]):
    name = "read_file"
    description = (
        "Read a UTF-8 text file, optionally a line range. Each line is shown as "
        '"<line number>: <text>"; the number and ": " are not part of the file, so leave '
        f"them out of edit_file's old_text. At most {_MAX_LINES} lines are shown per call; "
        "when lines are left out, the last line of the output says how to read on."
    )
    permission = "read"
    args_model = ReadFileArgs

    def run(self, parsed_args: ReadFileArgs, context: ToolContext) -> ToolResult:
        try:
            target = resolve_repo_path(root=context.repo_root, relative_path=parsed_args.path)
        except ValueError as exc:
            raise ToolExecutionError(str(exc)) from exc

        if not target.exists() or not target.is_file():
            raise ToolExecutionError(f"path does not exist or is not a file: {parsed_args.path}")
        if is_binary_file(target):
            raise ToolExecutionError(f"cannot read binary file: {parsed_args.path}")
        if parsed_args.end_line is not None and parsed_args.end_line < parsed_args.start_line:
            raise ToolExecutionError("end_line must be greater than or equal to start_line")

        start = parsed_args.start_line
        last = start + _MAX_LINES - 1
        if parsed_args.end_line is not None:
            last = min(last, parsed_args.end_line)
        shown: list[str] = []
        size = 0
        number = 0
        capped = more = False
        try:
            # The file is read only up to the requested range, and only the
            # lines shown are decoded.
            with target.open("rb") as handle:
                lines = _Lines(handle)
                number = lines.skip(start - 1)
                while (line := lines.next()) is not None:
                    number += 1
                    if number > last:
                        more = True
                        break
                    if number < start:
                        continue
                    entry = f"{number}: {_shown(*line)}"
                    size += len(entry.encode("utf-8")) + 1
                    if size > _MAX_OUTPUT_BYTES and shown:
                        capped = more = True
                        break
                    shown.append(entry)
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(f"line {number} is not valid UTF-8: {exc}") from exc

        if start > max(number, 1):
            raise ToolExecutionError(
                f"start_line {start} is past the end of the file ({number} lines)"
            )
        end = start + len(shown) - 1
        # A read that reaches the end of the file needs no note.
        footer = ""
        if more and parsed_args.end_line is not None and end >= parsed_args.end_line:
            footer = f"(Showing lines {start}-{end}; the file has more lines.)"
        elif more:
            footer = (
                f"({'Output capped. ' if capped else ''}Showing lines {start}-{end}; "
                f"the file has more lines. Use start_line={end + 1} to continue.)"
            )
        output = "\n".join([*shown, "", footer] if footer else shown)

        return ToolResult(
            id=context.call_id,
            name=self.name,
            state=ToolExecutionState.COMPLETED,
            ok=True,
            output=output,
            metadata={
                "path": str(target.relative_to(context.repo_root)),
                "start_line": start,
                "end_line": end,
                "line_count": len(shown),
            },
        )


_BREAK = re.compile(rb"\r\n|\r|\n")
_CHUNK = 65536
# A shown line is cut at _MAX_LINE_CHARS characters, at most 4 bytes each.
_MAX_LINE_BYTES = 4 * _MAX_LINE_CHARS


class _Lines:
    """The lines of a file, read a chunk at a time: a line ends at "\\n",
    "\\r\\n" or "\\r", as LineView splits them. Of a line longer than
    _MAX_LINE_BYTES only the start is kept; the rest is dropped as it is read."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._buffer = b""
        self._pos = 0  # where the next line starts in the buffer
        self._eof = False
        self._at_start = True

    def _fill(self) -> None:
        chunk = self._handle.read(_CHUNK)
        if self._at_start:
            # A byte order mark may arrive split over reads: wait for 3 bytes.
            while chunk and len(chunk) < 3 and b"\xef\xbb\xbf".startswith(chunk):
                more = self._handle.read(_CHUNK)
                if not more:
                    break
                chunk += more
        self._eof = not chunk
        if self._at_start:
            self._at_start = False
            chunk = chunk.removeprefix(b"\xef\xbb\xbf")
        self._buffer = self._buffer[self._pos :] + chunk
        self._pos = 0

    def skip(self, count: int) -> int:
        """Pass over up to `count` lines, counting line breaks a chunk at a
        time; return how many were passed."""
        passed = 0
        in_line = False  # text read since the last line break
        while passed < count:
            if self._pos >= len(self._buffer) and not self._eof:
                self._fill()
                continue
            rest = self._buffer[self._pos :]
            # A "\r" at the end may be the start of a "\r\n": decided later.
            waiting = not self._eof and rest.endswith(b"\r")
            whole = rest[:-1] if waiting else rest
            breaks = whole.count(b"\r") + whole.count(b"\n") - whole.count(b"\r\n")
            if passed + breaks >= count:
                for match in _BREAK.finditer(self._buffer, self._pos):
                    passed += 1
                    self._pos = match.end()
                    if passed == count:
                        return passed
            passed += breaks
            tail = len(whole) - max(whole.rfind(b"\r"), whole.rfind(b"\n")) - 1
            in_line = tail > 0 or (in_line and not breaks)
            if self._eof:
                # The last line may end without a line break.
                if in_line and passed < count:
                    passed += 1
                self._pos = len(self._buffer)
                return passed
            self._pos = len(self._buffer) - waiting
            self._fill()
        return passed

    def next(self) -> Optional[Tuple[bytes, bool]]:
        """The next line's bytes without its line break, and whether only its
        start was kept; None at the end of the file."""
        kept: Optional[bytes] = None
        while True:
            match = _BREAK.search(self._buffer, self._pos)
            # A "\r" at the end of what was read may be the start of "\r\n".
            split = match is not None and match.group() == b"\r" and match.end() == len(
                self._buffer
            )
            if match is not None and (not split or self._eof):
                line = self._buffer[self._pos : match.start()]
                self._pos = match.end()
                if kept is None and len(line) > _MAX_LINE_BYTES:
                    kept = line[:_MAX_LINE_BYTES]
                return (kept, True) if kept is not None else (line, False)
            if self._eof:
                line = self._buffer[self._pos :]
                self._pos = len(self._buffer)
                if kept is None and not line:
                    return None
                if kept is None and len(line) > _MAX_LINE_BYTES:
                    kept = line[:_MAX_LINE_BYTES]
                return (kept, True) if kept is not None else (line, False)
            # A "\r" waiting for its "\n" is not part of the line.
            if kept is None and len(self._buffer) - self._pos - split > _MAX_LINE_BYTES:
                kept = self._buffer[self._pos : self._pos + _MAX_LINE_BYTES]
            if kept is not None:  # drop the rest of the long line read so far
                self._buffer, self._pos = (b"\r" if split else b""), 0
            self._fill()


def _shown(line: bytes, cut: bool) -> str:
    """A line as shown, cut at _MAX_LINE_CHARS characters."""
    if cut:  # its kept start may end inside a character
        text = codecs.getincrementaldecoder("utf-8")().decode(line)
    else:
        text = line.decode("utf-8")
    if cut or len(text) > _MAX_LINE_CHARS:
        return f"{text[:_MAX_LINE_CHARS]}... (line cut at {_MAX_LINE_CHARS} characters)"
    return text
