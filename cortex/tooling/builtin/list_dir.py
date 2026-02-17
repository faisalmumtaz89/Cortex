"""Read-only list directory tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.builtin.common import iter_rel_entries, resolve_repo_path, walk_rel_entries
from cortex.tooling.types import ToolExecutionState, ToolResult


class ListDirArgs(BaseModel):
    path: str = Field(default=".")
    recursive: bool = Field(default=False)
    max_depth: int = Field(default=2, ge=0, le=20)
    max_entries: int = Field(default=200, ge=1, le=5000)


class ListDirTool(BaseTool[ListDirArgs]):
    name = "list_dir"
    description = "List files/directories within the repository root."
    permission = "list"
    args_model = ListDirArgs

    def run(self, parsed_args: ListDirArgs, context: ToolContext) -> ToolResult:
        try:
            target = resolve_repo_path(root=context.repo_root, relative_path=parsed_args.path)
        except ValueError as exc:
            raise ToolExecutionError(str(exc)) from exc

        if not target.exists():
            raise ToolExecutionError(f"path does not exist: {parsed_args.path}")
        if not target.is_dir():
            raise ToolExecutionError(f"path is not a directory: {parsed_args.path}")

        entries = []
        iterator = (
            walk_rel_entries(context.repo_root, target, max_depth=parsed_args.max_depth)
            if parsed_args.recursive
            else iter_rel_entries(context.repo_root, target)
        )
        for entry in iterator:
            entries.append(entry)
            if len(entries) >= parsed_args.max_entries:
                break

        return ToolResult(
            id=context.call_id,
            name=self.name,
            state=ToolExecutionState.COMPLETED,
            ok=True,
            output="\n".join(entries),
            metadata={"entries": entries, "count": len(entries)},
        )
