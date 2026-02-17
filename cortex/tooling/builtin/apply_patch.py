"""Patch application tool (unified-diff based)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.types import ToolExecutionState, ToolResult

_UNIFIED_PATH_RE = re.compile(r"^(?:--- a/|\+\+\+ b/)(.+)$")


class ApplyPatchArgs(BaseModel):
    patch_text: str = Field(min_length=1)


class ApplyPatchTool(BaseTool[ApplyPatchArgs]):
    name = "apply_patch"
    description = "Apply a unified git patch to repository files."
    permission = "edit"
    args_model = ApplyPatchArgs

    def run(self, parsed_args: ApplyPatchArgs, context: ToolContext) -> ToolResult:
        patch_text = parsed_args.patch_text
        touched_paths = self._extract_paths(patch_text, context.repo_root)
        if not touched_paths:
            raise ToolExecutionError("patch does not contain any file changes")

        check = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", "-"],
            cwd=context.repo_root,
            input=patch_text,
            text=True,
            capture_output=True,
        )
        if check.returncode != 0:
            error_line = (check.stderr or check.stdout or "git apply --check failed").strip()
            raise ToolExecutionError(error_line)

        apply_result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=context.repo_root,
            input=patch_text,
            text=True,
            capture_output=True,
        )
        if apply_result.returncode != 0:
            error_line = (apply_result.stderr or apply_result.stdout or "git apply failed").strip()
            raise ToolExecutionError(error_line)

        summary = "\n".join(sorted(touched_paths))
        return ToolResult(
            id=context.call_id,
            name=self.name,
            state=ToolExecutionState.COMPLETED,
            ok=True,
            output=f"Patch applied successfully.\n{summary}",
            metadata={"files": sorted(touched_paths)},
        )

    def _extract_paths(self, patch_text: str, repo_root: Path) -> set[str]:
        paths: set[str] = set()
        for line in patch_text.splitlines():
            match = _UNIFIED_PATH_RE.match(line)
            if not match:
                continue
            rel = match.group(1).strip()
            if rel == "/dev/null":
                continue
            if rel.startswith("/"):
                raise ToolExecutionError(f"absolute paths are not allowed in patch: {rel}")
            resolved = (repo_root / rel).resolve()
            if not resolved.is_relative_to(repo_root):
                raise ToolExecutionError(f"patch path escapes repository root: {rel}")
            paths.add(rel)
        return paths
