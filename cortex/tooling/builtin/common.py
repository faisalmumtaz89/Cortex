"""Helpers shared by builtin tooling implementations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def resolve_repo_path(*, root: Path, relative_path: str) -> Path:
    raw = str(relative_path or "").strip()
    if not raw:
        raise ValueError("path must be non-empty")
    if "\x00" in raw:
        raise ValueError("path contains null byte")
    if raw.startswith("~"):
        raise ValueError("path must be repo-relative")

    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError("path must be repo-relative")

    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("path escapes repo root")
    return resolved


def is_binary_file(path: Path, sniff_bytes: int = 4096) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(sniff_bytes)
    except OSError:
        return True
    return b"\x00" in chunk


def iter_rel_entries(root: Path, target: Path) -> Iterable[str]:
    for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
        rel = entry.relative_to(root)
        suffix = "/" if entry.is_dir() else ""
        yield f"{rel}{suffix}"


def walk_rel_entries(root: Path, target: Path, *, max_depth: int):
    base_depth = len(target.relative_to(root).parts)
    for dirpath, dirnames, filenames in os.walk(target):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue

        for dirname in sorted(dirnames):
            rel = (current / dirname).relative_to(root)
            yield f"{rel}/"

        for filename in sorted(filenames):
            rel = (current / filename).relative_to(root)
            yield str(rel)
