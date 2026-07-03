"""Whole-file write tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.builtin.common import is_binary_file, resolve_repo_path
from cortex.tooling.builtin.diff_util import build_line_diff
from cortex.tooling.types import ToolExecutionState, ToolResult

_MAX_WRITE_BYTES = 5_000_000


class WriteFileArgs(BaseModel):
    path: str
    content: str = Field(default="")


class WriteFileTool(BaseTool[WriteFileArgs]):
    name = "write_file"
    description = (
        "Create a file or overwrite an existing one with the given content. "
        "Parent directories are created as needed. Prefer edit_file for partial changes."
    )
    permission = "edit"
    args_model = WriteFileArgs

    def run(self, parsed_args: WriteFileArgs, context: ToolContext) -> ToolResult:
        try:
            target = resolve_repo_path(root=context.repo_root, relative_path=parsed_args.path)
        except ValueError as exc:
            raise ToolExecutionError(str(exc)) from exc

        if target.exists() and not target.is_file():
            raise ToolExecutionError(f"path exists and is not a file: {parsed_args.path}")
        if target.exists() and is_binary_file(target):
            raise ToolExecutionError(f"refusing to overwrite binary file: {parsed_args.path}")

        payload = parsed_args.content.encode("utf-8")
        if len(payload) > _MAX_WRITE_BYTES:
            raise ToolExecutionError(
                f"content too large ({len(payload)} bytes > {_MAX_WRITE_BYTES})"
            )

        existed = target.exists()
        # Capture the old content BEFORE overwriting so the diff is accurate.
        old_content = ""
        if existed:
            try:
                old_content = target.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                old_content = ""  # undecodable old file → skip the diff below

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(parsed_args.content, encoding="utf-8")

        action = "overwrote" if existed else "created"
        rel = target.relative_to(context.repo_root)
        rel_str = str(rel)
        metadata: dict = {"path": rel_str, "bytes": len(payload), "created": not existed}
        diff = build_line_diff(
            old_content,
            parsed_args.content,
            path=rel_str,
            op="create" if not existed else "overwrite",
            language=target.suffix.lstrip(".").lower() or None,
        )
        if diff:
            metadata["diff"] = diff

        return ToolResult(
            id=context.call_id,
            name=self.name,
            state=ToolExecutionState.COMPLETED,
            ok=True,
            output=f"{action} {rel} ({len(payload)} bytes)",
            metadata=metadata,
        )
