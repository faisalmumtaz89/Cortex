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
