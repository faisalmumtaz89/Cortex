"""Frontend sidecar launcher for Cortex OpenTUI runtime."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
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
    # -P: the worker runs in the user's project, which must never shadow the
    # modules Cortex imports.
    env["CORTEX_WORKER_ARGS"] = "-P -m cortex --worker-stdio"
    # The sidecar process runs from the repo/frontend dir; the worker (and so
    # the agent's tool sandbox) must run in the directory the user launched
    # Cortex from.
    env["CORTEX_PROJECT_DIR"] = os.getcwd()
    return env


# The compiled sidecar unpacks its native library into the temp folder at
# every start and never removes it. It gets a folder of its own, removed when
# it exits; the worker puts the user's TMPDIR back for the commands it runs.
_SIDECAR_TMP_PREFIX = "cortex-tui-"
USER_TMPDIR_VARIABLE = "CORTEX_USER_TMPDIR"


def _sidecar_tmpdir() -> Path:
    base = Path(tempfile.gettempdir())
    for stale in base.glob(f"{_SIDECAR_TMP_PREFIX}*"):
        # A folder whose launcher is gone (a crash, a killed terminal).
        pid = stale.name[len(_SIDECAR_TMP_PREFIX):].split("-", 1)[0]
        if pid.isdigit() and not _alive(int(pid)):
            shutil.rmtree(stale, ignore_errors=True)
    return Path(tempfile.mkdtemp(prefix=f"{_SIDECAR_TMP_PREFIX}{os.getpid()}-", dir=base))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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

    env = _build_worker_env()
    sidecar_tmp = _sidecar_tmpdir()
    env[USER_TMPDIR_VARIABLE] = os.environ.get("TMPDIR", "")
    env["TMPDIR"] = str(sidecar_tmp)
    try:
        return _run_sidecar(command, env)
    finally:
        shutil.rmtree(sidecar_tmp, ignore_errors=True)


def _run_sidecar(command: list[str], env: dict) -> int:
    process = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        env=env,
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
