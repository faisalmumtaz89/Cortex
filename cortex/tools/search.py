"""Search utilities for repo tools."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from cortex.tools.errors import ToolError, ValidationError
from cortex.tools.fs_ops import RepoFS


class RepoSearch:
    """Search helper constrained to a repo root."""

    def __init__(self, repo_fs: RepoFS) -> None:
        self.repo_fs = repo_fs

    def search(self, query: str, path: str = ".", use_regex: bool = True, max_results: int = 100) -> Dict[str, List[Dict[str, object]]]:
        if not isinstance(query, str) or not query:
            raise ValidationError("query must be a non-empty string")
        if max_results < 1:
            raise ValidationError("max_results must be >= 1")
        root = self.repo_fs.root
        target = self.repo_fs.resolve_path(path)

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
        pattern = re.compile(query) if use_regex else None
        results: List[Dict[str, object]] = []
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for name in filenames:
                path = Path(dirpath) / name
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                for idx, line in enumerate(text.splitlines(), start=1):
                    found = bool(pattern.search(line)) if pattern else (query in line)
                    if found:
                        results.append({"path": str(path.relative_to(self.repo_fs.root)), "line": idx, "text": line})
                        if len(results) >= max_results:
                            return results
        return results
