"""Filesystem operations scoped to a repo root."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cortex.tools.errors import ToolError, ValidationError


class RepoFS:
    """Filesystem helper constrained to a single repo root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def resolve_path(self, path: str) -> Path:
        if not path or not isinstance(path, str):
            raise ValidationError("path must be a non-empty string")
        raw = Path(path).expanduser()
        resolved = raw.resolve() if raw.is_absolute() else (self.root / raw).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValidationError(f"path escapes repo root ({self.root}); use a relative path like '.'")
        return resolved

    def list_dir(self, path: str = ".", recursive: bool = False, max_depth: int = 2, max_entries: int = 200) -> Dict[str, List[str]]:
        target = self.resolve_path(path)
        if not target.is_dir():
            raise ValidationError("path is not a directory")
        entries: List[str] = []
        if not recursive:
            for item in sorted(target.iterdir()):
                rel = item.relative_to(self.root)
                suffix = "/" if item.is_dir() else ""
                entries.append(f"{rel}{suffix}")
                if len(entries) >= max_entries:
                    break
            return {"entries": entries}

        base_depth = len(target.relative_to(self.root).parts)
        for dirpath, dirnames, filenames in os.walk(target):
            depth = len(Path(dirpath).relative_to(self.root).parts) - base_depth
            if depth > max_depth:
                dirnames[:] = []
                continue
            for name in sorted(dirnames):
                rel = (Path(dirpath) / name).relative_to(self.root)
                entries.append(f"{rel}/")
                if len(entries) >= max_entries:
                    return {"entries": entries}
            for name in sorted(filenames):
                rel = (Path(dirpath) / name).relative_to(self.root)
                entries.append(str(rel))
                if len(entries) >= max_entries:
                    return {"entries": entries}
        return {"entries": entries}

    def read_text(self, path: str, start_line: int = 1, end_line: Optional[int] = None, max_bytes: int = 2_000_000) -> Dict[str, object]:
        target = self.resolve_path(path)
        if not target.is_file():
            raise ValidationError("path is not a file")
        size = target.stat().st_size
        if size > max_bytes and start_line == 1 and end_line is None:
            raise ToolError("file too large; specify a line range")
        if start_line < 1:
            raise ValidationError("start_line must be >= 1")
        if end_line is not None and end_line < start_line:
            raise ValidationError("end_line must be >= start_line")

        lines: List[str] = []
        with target.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                if idx < start_line:
                    continue
                if end_line is not None and idx > end_line:
                    break
                lines.append(line.rstrip("\n"))
        content = "\n".join(lines)
        return {"path": str(target.relative_to(self.root)), "content": content, "start_line": start_line, "end_line": end_line}

    def read_full_text(self, path: str) -> str:
        target = self.resolve_path(path)
        if not target.is_file():
            raise ValidationError("path is not a file")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ToolError(f"file is not valid utf-8: {e}") from e

    def write_text(self, path: str, content: str, expected_sha256: Optional[str] = None) -> Dict[str, object]:
        target = self.resolve_path(path)
        if not target.exists() or not target.is_file():
            raise ValidationError("path does not exist or is not a file")
        if expected_sha256:
            current = self.read_full_text(path)
            if self.sha256_text(current) != expected_sha256:
                raise ToolError("file changed; expected hash does not match")
        target.write_text(content, encoding="utf-8")
        return {"path": str(target.relative_to(self.root)), "sha256": self.sha256_text(content)}

    def create_text(self, path: str, content: str, overwrite: bool = False) -> Dict[str, object]:
        target = self.resolve_path(path)
        if target.exists() and not overwrite:
            raise ValidationError("path already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target.relative_to(self.root)), "sha256": self.sha256_text(content)}

    def delete_file(self, path: str) -> Dict[str, object]:
        target = self.resolve_path(path)
        if not target.exists() or not target.is_file():
            raise ValidationError("path does not exist or is not a file")
        if not self._is_git_tracked(target):
            raise ToolError("delete blocked: file is not tracked by git")
        target.unlink()
        return {"path": str(target.relative_to(self.root)), "deleted": True}

    def sha256_text(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _is_git_tracked(self, target: Path) -> bool:
        git_dir = self.root / ".git"
        if not git_dir.exists():
            return False
        rel = str(target.relative_to(self.root))
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
