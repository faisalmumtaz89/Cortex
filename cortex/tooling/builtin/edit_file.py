"""Exact string-replacement edit tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.builtin.common import is_binary_file, resolve_repo_path
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

    def run(self, parsed_args: EditFileArgs, context: ToolContext) -> ToolResult:
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
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(f"file is not valid UTF-8: {exc}") from exc

        occurrences = content.count(parsed_args.old_text)
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
        if parsed_args.replace_all:
            updated = content.replace(parsed_args.old_text, parsed_args.new_text)
        else:
            updated = content.replace(parsed_args.old_text, parsed_args.new_text, 1)
        target.write_text(updated, encoding="utf-8")

        rel_path = str(target.relative_to(context.repo_root))
        metadata: dict = {"path": rel_path, "replacements": replaced}
        # Structured diff for the TUI's green/red renderer. Diffing the FULL
        # old vs new content (not just old_text/new_text) keeps line numbers
        # absolute and handles replace_all correctly.
        diff = build_line_diff(
            content,
            updated,
            path=rel_path,
            op="edit",
            language=target.suffix.lstrip(".").lower() or None,
        )
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
