"""Runtime IO safety guards for redirected stdio."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def _parse_size_to_bytes(raw: str, default_bytes: int) -> int:
    if not raw:
        return default_bytes

    value = str(raw).strip().lower()
    units = (
        ("gb", 1024 * 1024 * 1024),
        ("mb", 1024 * 1024),
        ("kb", 1024),
        ("b", 1),
    )

    for suffix, multiplier in units:
        if value.endswith(suffix):
            number = value[: -len(suffix)].strip()
            try:
                return max(1, int(float(number) * multiplier))
            except Exception:
                return default_bytes

    try:
        return max(1, int(value))
    except Exception:
        return default_bytes


def _resolve_regular_file_for_fd(fd: int) -> Path | None:
    def _candidate_path() -> Path | None:
        try:
            import fcntl  # type: ignore

            if hasattr(fcntl, "F_GETPATH"):
                raw = fcntl.fcntl(fd, fcntl.F_GETPATH, b"\0" * 1024)
                if isinstance(raw, bytes):
                    decoded = raw.split(b"\0", 1)[0].decode(errors="ignore")
                else:
                    decoded = str(raw).split("\0", 1)[0]
                if decoded:
                    return Path(decoded)
        except Exception:
            pass

        for probe in (f"/proc/self/fd/{fd}", f"/dev/fd/{fd}"):
            try:
                target = os.readlink(probe)
                if target:
                    return Path(target)
            except Exception:
                continue
        return None

    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
    except Exception:
        return None

    path = _candidate_path()
    if path is None:
        return None

    try:
        if path.exists() and path.is_file():
            return path.resolve()
    except Exception:
        return None
    return None


def _trim_file_to_tail(path: Path, *, max_bytes: int, tail_bytes: int) -> None:
    try:
        size = path.stat().st_size
        if size <= max_bytes:
            return

        keep = max(1, min(tail_bytes, max_bytes))
        with path.open("rb+") as handle:
            if size > keep:
                handle.seek(size - keep)
            tail = handle.read(keep)
            handle.seek(0)
            handle.write(tail)
            handle.truncate(len(tail))
    except Exception:
        # Guard path must never break user flow.
        return


def bound_redirected_stdio_files() -> None:
    """Keep redirected stdout/stderr regular files bounded in size."""
    max_bytes = _parse_size_to_bytes(
        os.environ.get("CORTEX_STDIO_FILE_MAX_SIZE", "256MB"),
        256 * 1024 * 1024,
    )
    tail_bytes = _parse_size_to_bytes(
        os.environ.get("CORTEX_STDIO_FILE_TAIL_SIZE", "1MB"),
        1 * 1024 * 1024,
    )

    seen: set[Path] = set()
    for fd in (1, 2):
        path = _resolve_regular_file_for_fd(fd)
        if path is None or path in seen:
            continue
        seen.add(path)
        _trim_file_to_tail(path, max_bytes=max_bytes, tail_bytes=tail_bytes)
