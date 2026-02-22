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


def _bun_dev_command(bun: str) -> list[str]:
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


def _prefer_dev_entrypoint(*, bun: Optional[str], entry: Path) -> bool:
    if bun is None or not entry.exists():
        return False

    if os.environ.get("CORTEX_TUI_FORCE_BUNDLED") == "1":
        return False
    if os.environ.get("CORTEX_TUI_PREFER_DEV") == "1":
        return True

    # In a source checkout, prefer live TS entrypoint when Bun is available
    # to avoid stale precompiled sidecar binaries during iterative development.
    return (_repo_root() / ".git").exists()


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
    if env.get("CORTEX_PRESERVE_OTUI_ENV") != "1":
        for key in list(env):
            if key.startswith("OTUI_"):
                env.pop(key, None)
    env["CORTEX_WORKER_CMD"] = sys.executable
    env["CORTEX_WORKER_ARGS"] = "-m cortex --worker-stdio"
    return env


def _candidate_command() -> Optional[list[str]]:
    bundled = _bundled_binary()
    bun = _find_bun()
    entry = _dev_entrypoint()

    if _prefer_dev_entrypoint(bun=bun, entry=entry):
        if bun is None:
            return None
        return _bun_dev_command(bun)

    if bundled.exists() and os.access(bundled, os.X_OK):
        # In repository/dev mode this can be a shell wrapper that still requires Bun.
        # Only run script wrappers if Bun is available; run true bundled binaries always.
        if not _is_script_file(bundled) or bun:
            return [str(bundled)]

    if not entry.exists():
        return None

    if bun:
        return _bun_dev_command(bun)

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
    try:
        return process.wait()
    except KeyboardInterrupt:
        # Ensure the child is reaped before bubbling Ctrl-C to the caller.
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        raise
