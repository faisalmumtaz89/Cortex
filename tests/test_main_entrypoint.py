from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_main_exits_130_on_keyboard_interrupt_without_traceback() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        "from cortex import __main__ as cli\n"
        "def _raise_interrupt():\n"
        "    raise KeyboardInterrupt()\n"
        "cli.launch_tui = _raise_interrupt\n"
        "cli._has_terminal = lambda: True\n"
        "cli.main()\n"
    )

    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 130
    assert "Traceback" not in process.stderr


def test_the_app_without_a_terminal_says_to_use_headless_mode(tmp_path: Path) -> None:
    """Run from a script or a pipe, the interactive app would draw into
    nothing and wait for keys that never come."""
    repo_root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, "-P", "-m", "cortex"],
        cwd=tmp_path,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo_root)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 2, process.stderr
    assert "needs a terminal" in process.stderr
    assert "cortex -p" in process.stderr


def test_cortex_keeps_its_home_folder_private(tmp_path: Path) -> None:
    """~/.cortex holds conversations, logs and settings: only its owner may
    read it, also when the installer created it with the usual 0755."""
    repo_root = Path(__file__).resolve().parents[1]
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo_root)}
    folder = tmp_path / ".cortex"
    folder.mkdir(mode=0o755)
    folder.chmod(0o755)
    subprocess.run([sys.executable, "-P", "-m", "cortex", "--version"], env=env, check=True,
                   capture_output=True, timeout=60)
    assert folder.stat().st_mode & 0o777 == 0o700

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    subprocess.run([sys.executable, "-P", "-m", "cortex", "--version"],
                   env={**env, "HOME": str(fresh)}, check=True, capture_output=True, timeout=60)
    assert (fresh / ".cortex").stat().st_mode & 0o777 == 0o700
