"""Helpers shared by builtin tooling implementations."""

from __future__ import annotations

import bisect
import ctypes
import errno
import os
import re
import stat
import sys
import uuid
from pathlib import Path
from typing import Iterable, Optional

from cortex.tooling.base import ToolExecutionError


def resolve_repo_path(*, root: Path, relative_path: str) -> Path:
    raw = str(relative_path or "").strip()
    if not raw:
        raise ValueError("path must be non-empty")
    if "\x00" in raw:
        raise ValueError("path contains null byte")
    if raw.startswith("~"):
        raise ValueError("path must be inside the working directory")

    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"path is outside the working directory {root}; file tools are "
            "sandboxed to it (use bash for files elsewhere)"
        )
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


_NOT_WALKED = frozenset({".git"})


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
        # Named, not walked: git's own objects and hooks.
        dirnames[:] = [name for name in dirnames if name not in _NOT_WALKED]

        for filename in sorted(filenames):
            rel = (current / filename).relative_to(root)
            yield str(rel)


LINE_BREAK = re.compile(r"\r\n?|\n")


def line_break(text: str) -> str:
    """The line break `text` uses most: "\\r\\n", "\\r" or "\\n" (also for text
    without one)."""
    crlf = text.count("\r\n")
    counts = {"\n": text.count("\n") - crlf, "\r\n": crlf, "\r": text.count("\r") - crlf}
    best = max(counts, key=lambda brk: counts[brk])
    return best if counts[best] else "\n"


class LineView:
    """Text as read_file shows it, every line break as "\\n", with the way back
    to positions in the text itself, so an edit made on the view changes only
    the part of the text it covers."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.view = LINE_BREAK.sub("\n", text)
        # Positions in the view of the "\n" standing for a two-character "\r\n".
        self._crlf: list[int] = []
        for number, match in enumerate(re.finditer("\r\n", text)):
            self._crlf.append(match.start() - number)

    def position(self, index: int) -> int:
        """Where position `index` of the view is in the text."""
        return index + bisect.bisect_left(self._crlf, index)


def view_of(text: str) -> str:
    """`text` with every line break as "\\n"."""
    return LINE_BREAK.sub("\n", text)


def write_text_atomic(path: Path, text: str) -> None:
    """Write `text` as UTF-8 so that `path` holds either its old or its new
    content, never a part: the new content goes to a private temporary file in
    the same directory, which then replaces `path` and keeps its mode, owner,
    group, extended attributes and ACLs. A read-only file is refused. Where
    the file cannot be replaced keeping all of that (hard links, an owner or
    attribute this process cannot set, a file mounted on its own, a directory
    that does not allow new files), it is written in place."""
    data = text.encode("utf-8")
    try:
        st: Optional[os.stat_result] = os.stat(path)
    except FileNotFoundError:
        st = None
    if st is not None and not os.access(path, os.W_OK):
        raise ToolExecutionError(f"file is read-only: {path.name}")
    if st is not None and (
        st.st_nlink > 1 or (st.st_uid != os.geteuid() and os.geteuid() != 0)
    ):
        path.write_bytes(data)  # replacing would cut the other links, or change the owner
        return
    partial = path.parent / f".cortex-{uuid.uuid4().hex[:8]}.tmp"
    # A new file is created as any file is (the umask, or the directory's
    # default ACL, decides its permissions); the copy of an existing file is
    # private until it has the old file's mode.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        fd = os.open(partial, flags, 0o600 if st is not None else 0o666)
    except PermissionError:
        path.write_bytes(data)
        return
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
            kept = st is None or _keep_metadata(out.fileno(), path, partial, st)
            if st is not None:
                os.fchmod(out.fileno(), stat.S_IMODE(st.st_mode))
        if not kept:
            partial.unlink(missing_ok=True)
            path.write_bytes(data)
            return
        try:
            os.replace(partial, path)
        except OSError as exc:
            if exc.errno not in (errno.EBUSY, errno.EXDEV, errno.EPERM):
                raise
            partial.unlink(missing_ok=True)
            path.write_bytes(data)
            return
        _sync_directory(path.parent)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _keep_metadata(fd: int, path: Path, partial: Path, st: os.stat_result) -> bool:
    """Give the new file the old one's owner, group, extended attributes and
    ACLs; False when that is not possible."""
    try:
        if (st.st_uid, st.st_gid) != (os.geteuid(), os.getegid()):
            os.fchown(fd, st.st_uid, st.st_gid)
        if sys.platform == "linux":  # ACLs are system.posix_acl_* attributes
            try:
                names = os.listxattr(path)
            except OSError as exc:
                if exc.errno != errno.ENOTSUP:
                    raise
                return True  # the filesystem has no extended attributes
            for name in names:
                os.setxattr(fd, name, os.getxattr(path, name))
            # The new file inherits the directory's default ACL; the old one
            # may have had none.
            for name in set(os.listxattr(fd)) - set(names):
                os.removexattr(fd, name)
        elif sys.platform == "darwin":
            _copy_darwin_metadata(path, partial)
    except OSError:
        return False
    return True


def _copy_darwin_metadata(path: Path, partial: Path) -> None:
    """Copy extended attributes and ACLs with macOS's copyfile(3)."""
    copyfile_acl, copyfile_xattr = 1 << 0, 1 << 2
    libc = ctypes.CDLL("libc.dylib", use_errno=True)
    libc.copyfile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint32]
    if libc.copyfile(bytes(path), bytes(partial), None, copyfile_acl | copyfile_xattr) != 0:
        raise OSError(ctypes.get_errno(), "copyfile failed")


def _sync_directory(directory: Path) -> None:
    """Make the replacement itself survive a crash."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
