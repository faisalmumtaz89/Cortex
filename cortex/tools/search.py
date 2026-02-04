"""Search utilities for repo tools."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from cortex.tools.errors import ToolError, ValidationError
from cortex.tools.fs_ops import RepoFS


class RepoSearch:
    """Search helper constrained to a repo root."""

    _DEFAULT_SKIP_DIRS = {
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

    def __init__(self, repo_fs: RepoFS) -> None:
        self.repo_fs = repo_fs

    def search(self, query: str, path: str = ".", use_regex: bool = True, max_results: int = 100) -> Dict[str, List[Dict[str, object]]]:
        if not isinstance(query, str) or not query:
            raise ValidationError("query must be a non-empty string")
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise ValidationError("max_results must be an int")
        if max_results < 1:
            raise ValidationError("max_results must be >= 1")
        if not isinstance(use_regex, bool):
            raise ValidationError("use_regex must be a bool")
        target = self.repo_fs.resolve_path(path)
        if not target.exists():
            raise ValidationError("path does not exist")

        if shutil.which("rg"):
            return {"results": self._rg_search(query, target, use_regex, max_results)}
        return {"results": self._python_search(query, target, use_regex, max_results)}

    def _rg_search(self, query: str, target: Path, use_regex: bool, max_results: int) -> List[Dict[str, object]]:
        args = ["rg", "--line-number", "--with-filename", "--no-heading"]
        if not use_regex:
            args.append("-F")
        args.extend(["-e", query, str(target)])
        result = subprocess.run(args, cwd=self.repo_fs.root, capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise ToolError(f"rg failed: {result.stderr.strip()}")
        matches: List[Dict[str, object]] = []
        for line in result.stdout.splitlines():
            try:
                file_path, line_no, text = line.split(":", 2)
            except ValueError:
                continue
            matches.append({"path": file_path, "line": int(line_no), "text": text})
            if len(matches) >= max_results:
                break
        return matches

    def _python_search(self, query: str, target: Path, use_regex: bool, max_results: int) -> List[Dict[str, object]]:
        pattern: Optional[re.Pattern[str]] = None
        if use_regex:
            try:
                pattern = re.compile(query)
            except re.error as e:
                raise ValidationError(f"invalid regex: {e}") from e
        results: List[Dict[str, object]] = []

        if target.is_file():
            if self._looks_binary(target):
                return results
            self._scan_file(target, pattern, query, results, max_results)
            return results

        skip_dirs = set(self._DEFAULT_SKIP_DIRS)
        if target.name in skip_dirs:
            skip_dirs.remove(target.name)

        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for name in filenames:
                path = Path(dirpath) / name
                if self._looks_binary(path):
                    continue
                if self._scan_file(path, pattern, query, results, max_results):
                    return results
        return results

    def _scan_file(
        self,
        path: Path,
        pattern: Optional[re.Pattern[str]],
        query: str,
        results: List[Dict[str, object]],
        max_results: int,
    ) -> bool:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for idx, line in enumerate(handle, start=1):
                    found = bool(pattern.search(line)) if pattern else (query in line)
                    if found:
                        results.append(
                            {
                                "path": str(path.relative_to(self.repo_fs.root)),
                                "line": idx,
                                "text": line.rstrip("\n"),
                            }
                        )
                        if len(results) >= max_results:
                            return True
        except OSError:
            return False
        return False

    def _looks_binary(self, path: Path, sniff_bytes: int = 4096) -> bool:
        try:
            with path.open("rb") as handle:
                chunk = handle.read(sniff_bytes)
        except OSError:
            return True
        return b"\x00" in chunk
