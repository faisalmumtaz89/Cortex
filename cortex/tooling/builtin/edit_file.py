"""Exact string-replacement edit tool."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.builtin.common import (
    LINE_BREAK,
    LineView,
    is_binary_file,
    line_break,
    resolve_repo_path,
    view_of,
    write_text_atomic,
)
from cortex.tooling.builtin.diff_util import build_line_diff
from cortex.tooling.types import ToolExecutionState, ToolResult


class EditFileArgs(BaseModel):
    path: str
    old_text: str = Field(min_length=1)
    new_text: str
    replace_all: bool = False


class EditFileTool(BaseTool[EditFileArgs]):
    name = "edit_file"
    description = (
        "Replace an exact text snippet in a file. old_text must match the file contents "
        "exactly (including whitespace) and exactly once, unless replace_all is true."
    )
    permission = "edit"
    args_model = EditFileArgs

    def _plan(
        self, parsed_args: EditFileArgs, context: ToolContext
    ) -> tuple[Path, LineView, str, int]:
        """The file, its text as read, its text after the edit, and the number of
        replacements; ToolExecutionError when the edit cannot be made."""
        try:
            target = resolve_repo_path(root=context.repo_root, relative_path=parsed_args.path)
        except ValueError as exc:
            raise ToolExecutionError(str(exc)) from exc

        if not target.exists() or not target.is_file():
            raise ToolExecutionError(f"path does not exist or is not a file: {parsed_args.path}")
        if is_binary_file(target):
            raise ToolExecutionError(f"cannot edit binary file: {parsed_args.path}")
        if parsed_args.old_text == parsed_args.new_text:
            raise ToolExecutionError("new_text must differ from old_text")

        try:
            content = target.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(f"file is not valid UTF-8: {exc}") from exc

        # old_text and new_text are matched against the file as read_file
        # shows it, every line break as "\n"; the edit then changes only the
        # part of the file it covers, so other lines keep their own line breaks.
        # read_file never shows a "\r", so one in the arguments is dropped.
        lines = LineView(content)
        old_text = parsed_args.old_text.replace("\r", "")
        new_text = parsed_args.new_text.replace("\r", "")
        if not old_text:
            raise ToolExecutionError("old_text has no text besides line breaks")
        if old_text == new_text:
            raise ToolExecutionError(
                "old_text and new_text differ only in line breaks, which edit_file keeps "
                "as the file has them; to change a file's line breaks, use bash"
            )

        found: list[int] = []
        at = lines.view.find(old_text)
        while at != -1:
            found.append(at)
            at = lines.view.find(old_text, at + len(old_text))
        occurrences = len(found)
        if occurrences == 0:
            raise ToolExecutionError(
                "old_text not found in file; re-read the file and match its contents exactly"
            )
        if occurrences > 1 and not parsed_args.replace_all:
            raise ToolExecutionError(
                f"old_text matches {occurrences} locations; include more surrounding context "
                "to make it unique, or set replace_all to true"
            )

        replaced = occurrences if parsed_args.replace_all else 1
        expected = new_text.join(lines.view.split(old_text, replaced))
        updated = _replace(content, lines, found[:replaced], len(old_text), new_text, False)
        if view_of(updated) != expected:
            # A "\r" break followed by a "\n" one read as one "\r\n": use
            # "\r\n" for the replacement's breaks, which joins with no other.
            updated = _replace(content, lines, found[:replaced], len(old_text), new_text, True)
        if view_of(updated) != expected:
            # Removing all text between a "\r" and a "\n" joins the two.
            raise ToolExecutionError(
                "this edit would join a \\r and a \\n line break into one; keep some text "
                "between them, or make the change with bash"
            )
        return target, lines, updated, replaced

    def preview(self, parsed_args: EditFileArgs, context: ToolContext) -> Optional[dict]:
        target, lines, updated, _replaced = self._plan(parsed_args, context)
        return _diff(target, lines, updated, context)

    def run(self, parsed_args: EditFileArgs, context: ToolContext) -> ToolResult:
        target, lines, updated, replaced = self._plan(parsed_args, context)
        write_text_atomic(target, updated)

        rel_path = str(target.relative_to(context.repo_root))
        metadata: dict = {"path": rel_path, "replacements": replaced}
        diff = _diff(target, lines, updated, context)
        if diff:
            metadata["diff"] = diff

        return ToolResult(
            id=context.call_id,
            name=self.name,
            state=ToolExecutionState.COMPLETED,
            ok=True,
            output=f"edited {rel_path} ({replaced} replacement(s))",
            metadata=metadata,
        )


def _diff(target: Path, lines: LineView, updated: str, context: ToolContext) -> Optional[dict]:
    """Structured diff for the TUI's green/red renderer. Diffing the FULL old
    vs new content (not just old_text/new_text) keeps line numbers absolute and
    handles replace_all correctly."""
    return build_line_diff(
        lines.view,
        view_of(updated),
        path=str(target.relative_to(context.repo_root)),
        op="edit",
        language=target.suffix.lstrip(".").lower() or None,
    )


def _replace(
    content: str, lines: LineView, found: list[int], length: int, new_text: str, crlf: bool
) -> str:
    """`content` with the view's text of `length` at each of `found` replaced
    by `new_text`. Its line breaks are those of the text it replaces, in
    order, when it has as many, otherwise the file's usual one; with `crlf`,
    every one is "\\r\\n"."""
    usual = line_break(content)
    new_parts = new_text.split("\n")
    pieces: list[str] = []
    done = 0
    for at in found:
        start, stop = lines.position(at), lines.position(at + length)
        breaks = LINE_BREAK.findall(content, start, stop)
        if crlf or len(breaks) != len(new_parts) - 1:
            breaks = ["\r\n" if crlf else usual] * (len(new_parts) - 1)
        inserted = new_parts[0] + "".join(b + part for b, part in zip(breaks, new_parts[1:]))
        pieces += [content[done:start], inserted]
        done = stop
    return "".join([*pieces, content[done:]])
