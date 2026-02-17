"""Read-only content search tool."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.builtin.common import is_binary_file, resolve_repo_path
from cortex.tooling.types import ToolExecutionState, ToolResult

_SKIP_DIRS = {
    ".git",
    ".cortex",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


class SearchArgs(BaseModel):
    query: str = Field(min_length=1)
    path: str = Field(default=".")
    use_regex: bool = Field(default=True)
    max_results: int = Field(default=100, ge=1, le=5000)


class SearchTool(BaseTool[SearchArgs]):
    name = "search"
    description = "Search file contents in the repository (prefers ripgrep)."
    permission = "grep"
    args_model = SearchArgs

    def run(self, parsed_args: SearchArgs, context: ToolContext) -> ToolResult:
        try:
            target = resolve_repo_path(root=context.repo_root, relative_path=parsed_args.path)
        except ValueError as exc:
            raise ToolExecutionError(str(exc)) from exc

        if not target.exists():
            raise ToolExecutionError(f"path does not exist: {parsed_args.path}")

        if parsed_args.use_regex:
            try:
                re.compile(parsed_args.query)
            except re.error as exc:
                raise ToolExecutionError(f"invalid regex: {exc}") from exc

        results = self._rg_search(parsed_args, context.repo_root, target)
        if results is None:
            results = self._python_search(parsed_args, context.repo_root, target)

        lines = [f"{item['path']}:{item['line']}: {item['text']}" for item in results]
        return ToolResult(
            id=context.call_id,
            name=self.name,
            state=ToolExecutionState.COMPLETED,
            ok=True,
            output="\n".join(lines),
            metadata={"results": results, "count": len(results)},
        )

    def _rg_search(self, args: SearchArgs, root: Path, target: Path):
        if shutil.which("rg") is None:
            return None

        cmd = ["rg", "--line-number", "--with-filename", "--no-heading"]
        if not args.use_regex:
            cmd.append("-F")
        cmd.extend(["-e", args.query, str(target)])

        result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise ToolExecutionError(f"rg failed: {result.stderr.strip()}")

        matches = []
        for line in result.stdout.splitlines():
            try:
                file_path, line_no, text = line.split(":", 2)
            except ValueError:
                continue
            matches.append({"path": file_path, "line": int(line_no), "text": text})
            if len(matches) >= args.max_results:
                break
        return matches

    def _python_search(self, args: SearchArgs, root: Path, target: Path):
        pattern = re.compile(args.query) if args.use_regex else None
        matches = []

        def scan_file(path: Path) -> bool:
            if is_binary_file(path):
                return False
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line_no, line in enumerate(handle, start=1):
                        found = bool(pattern.search(line)) if pattern else (args.query in line)
                        if found:
                            matches.append(
                                {
                                    "path": str(path.relative_to(root)),
                                    "line": line_no,
                                    "text": line.rstrip("\n"),
                                }
                            )
                            if len(matches) >= args.max_results:
                                return True
            except OSError:
                return False
            return False

        if target.is_file():
            scan_file(target)
            return matches

        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
            for filename in filenames:
                if scan_file(Path(dirpath) / filename):
                    return matches
        return matches
