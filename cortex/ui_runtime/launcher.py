"""Frontend sidecar launcher for Cortex OpenTUI runtime."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bundled_binary() -> Path:
    return Path(__file__).resolve().parent / "bin" / "cortex-tui"


def _dev_entrypoint() -> Path:
    return _repo_root() / "frontend" / "cortex-tui" / "src" / "index.tsx"


def _frontend_project_dir() -> Path:
    return _repo_root() / "frontend" / "cortex-tui"


def _is_script_file(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(2) == b"#!"
    except OSError:
        return False


def _find_bun() -> Optional[str]:
    global_bun = shutil.which("bun")
    if global_bun:
        return global_bun

    frontend_dir = _frontend_project_dir()
    local_candidates = [
        frontend_dir / "node_modules" / ".bin" / "bun",
        frontend_dir / "node_modules" / "bun" / "bin" / "bun",
        frontend_dir / "node_modules" / "bun" / "bin" / "bun.exe",
    ]
    for candidate in local_candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _build_worker_env() -> dict:
    env = dict(os.environ)
    env["CORTEX_WORKER_CMD"] = sys.executable
    env["CORTEX_WORKER_ARGS"] = "-m cortex --worker-stdio"
    return env


def _candidate_command() -> Optional[list[str]]:
    bundled = _bundled_binary()
    bun = _find_bun()

    if bundled.exists() and os.access(bundled, os.X_OK):
        # In repository/dev mode this can be a shell wrapper that still requires Bun.
        # Only run script wrappers if Bun is available; run true bundled binaries always.
        if not _is_script_file(bundled) or bun:
            return [str(bundled)]

    entry = _dev_entrypoint()
    if not entry.exists():
        return None

    if bun:
        return [
            bun,
            "run",
            "--cwd",
            str(_frontend_project_dir()),
            "--preload",
            "@opentui/solid/preload",
            "--conditions=browser",
            "src/index.tsx",
        ]

    return None


def launch_tui() -> int:
    """Launch the frontend sidecar and return its exit code."""
    command = _candidate_command()
    if not command:
        return 127

    process = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        env=_build_worker_env(),
    )
    return process.wait()
