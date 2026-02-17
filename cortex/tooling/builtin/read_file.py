"""Read-only file read tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.builtin.common import is_binary_file, resolve_repo_path
from cortex.tooling.types import ToolExecutionState, ToolResult


class ReadFileArgs(BaseModel):
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    max_bytes: int = Field(default=2_000_000, ge=1, le=20_000_000)


class ReadFileTool(BaseTool[ReadFileArgs]):
    name = "read_file"
    description = "Read a UTF-8 text file by optional line range."
    permission = "read"
    args_model = ReadFileArgs

    def run(self, parsed_args: ReadFileArgs, context: ToolContext) -> ToolResult:
        try:
            target = resolve_repo_path(root=context.repo_root, relative_path=parsed_args.path)
        except ValueError as exc:
            raise ToolExecutionError(str(exc)) from exc

        if not target.exists() or not target.is_file():
            raise ToolExecutionError(f"path does not exist or is not a file: {parsed_args.path}")

        if target.stat().st_size > parsed_args.max_bytes and parsed_args.start_line == 1 and parsed_args.end_line is None:
            raise ToolExecutionError(
                f"file too large ({target.stat().st_size} bytes); provide a line range"
            )

        if is_binary_file(target):
            raise ToolExecutionError(f"cannot read binary file: {parsed_args.path}")

        if parsed_args.end_line is not None and parsed_args.end_line < parsed_args.start_line:
            raise ToolExecutionError("end_line must be greater than or equal to start_line")

        lines: list[str] = []
        try:
            with target.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle, start=1):
                    if idx < parsed_args.start_line:
                        continue
                    if parsed_args.end_line is not None and idx > parsed_args.end_line:
                        break
                    lines.append(line.rstrip("\n"))
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(f"file is not valid UTF-8: {exc}") from exc

        output = "\n".join(lines)
        return ToolResult(
            id=context.call_id,
            name=self.name,
            state=ToolExecutionState.COMPLETED,
            ok=True,
            output=output,
            metadata={
                "path": str(target.relative_to(context.repo_root)),
                "start_line": parsed_args.start_line,
                "end_line": parsed_args.end_line,
                "line_count": len(lines),
            },
        )
